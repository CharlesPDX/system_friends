import argparse
import json
import math
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import system_one_model
import system_two_model
from app_graph import create_system_two_node_graph
from config import SystemConfiguration
from experiment_model import SystemOnePrompt, SystemOneResponse
from history import create_database_and_table, record_interaction
from metacognitive import (
    MetacognitiveActivationComputation,
    MetacognitiveVector,
    MetacognitiveVectorComputation,
)
from orchestrator import Orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    await reset_system()

    yield
    # cleanup/shutdown goes here, if necessary


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    response, id = await run_system_one(user_input)
    return f"""
<div class="message is-bot" 
     hx-get="/get_chart?id={id}" 
     hx-trigger="click" 
     hx-target="#system-details">
        <div class="message-body">Bot: {response}</div>
</div>"""


msv_state: defaultdict[str, list[MetacognitiveVector]] = defaultdict(list)
system_two_state: dict[str, system_two_model.SystemTwoResponse] = {}
selected_nodes: list[system_two_model.NodeResponse] = []


class ChartNames(StrEnum):
    overall_msv = "Overall MSV"
    emotional_response = "Emotional Response"
    correctness = "Correctness"
    experiential_matching = "Experiential Matching"
    conflict_information = "Conflict Information"
    problem_importance = "Problem Importance"


@app.get("/get_chart", response_class=HTMLResponse)
async def get_chart(request: Request, id: str | None = None):
    msv_response = []
    msv_graphs = []
    msv_bar_graphs = []
    system_two_graph_components = ("", "")

    if id:
        for system_number, msv in enumerate(msv_state.get(id, [])):
            system_label = f"System {system_number+1}"
            msv_response.append(
                json.dumps(
                    asdict(msv)
                    | {
                        "activation_result": MetacognitiveActivationComputation.get_activation_result(
                            system_configuration.activation_computation_key,
                            msv,
                            system_configuration.additional_configuration,
                        )
                    },
                    indent=2,
                )
            )

            data = {
                "emotional_response": msv.emotional_response.calculated_value,
                "correctness": msv.correctness.calculated_value,
                "experiential_matching": msv.experiential_matching.calculated_value,
                "conflict_information": msv.conflict_information.calculated_value,
                "problem_importance": msv.problem_importance.calculated_value,
            }
            emotional_data = _clean_values(msv.emotional_response)
            correctness_data = _clean_values(msv.correctness)
            experiential_matching_data = _clean_values(msv.experiential_matching)
            conflict_information_data = _clean_values(msv.conflict_information)
            problem_importance_data = _clean_values(msv.problem_importance)

            # Create a Bokeh plot
            msv_components_chart = _generate_chart(
                data, "MSV Components", f"{system_label} MSV"
            )
            emotion_chart = _generate_chart(
                emotional_data, "Emotion Components", f"{system_label} Emotion Vector"
            )
            correctness_chart = _generate_chart(
                correctness_data,
                "Correctness Components",
                f"{system_label} Correctness Vector",
            )
            experiential_chart = _generate_chart(
                experiential_matching_data,
                "Experiential Components",
                f"{system_label} Experiential Vector",
            )
            conflict_chart = _generate_chart(
                conflict_information_data,
                "Conflict Components",
                f"{system_label} Conflict Vector",
            )
            problem_importance_chart = _generate_chart(
                problem_importance_data,
                "Problem Importance Components",
                f"{system_label} Problem Importance Vector",
            )

            bar_msv_components_chart = _generate_bar_chart(
                data, "MSV Components", f"{system_label} MSV"
            )
            bar_emotion_chart = _generate_bar_chart(
                emotional_data, "Emotion Components", f"{system_label} Emotion Vector"
            )
            bar_correctness_chart = _generate_bar_chart(
                correctness_data,
                "Correctness Components",
                f"{system_label} Correctness Vector",
            )
            bar_experiential_chart = _generate_bar_chart(
                experiential_matching_data,
                "Experiential Components",
                f"{system_label} Experiential Vector",
            )
            bar_conflict_chart = _generate_bar_chart(
                conflict_information_data,
                "Conflict Components",
                f"{system_label} Conflict Vector",
            )
            bar_problem_importance_chart = _generate_bar_chart(
                problem_importance_data,
                "Problem Importance Components",
                f"{system_label} Problem Importance Vector",
            )

            # Generate the plots' HTML
            parts = components(
                {
                    ChartNames.overall_msv.value: msv_components_chart,
                    ChartNames.emotional_response.value: emotion_chart,
                    ChartNames.correctness.value: correctness_chart,
                    ChartNames.experiential_matching.value: experiential_chart,
                    ChartNames.conflict_information.value: conflict_chart,
                    ChartNames.problem_importance.value: problem_importance_chart,
                }
            )
            bar_parts = components(
                {
                    ChartNames.overall_msv.value: bar_msv_components_chart,
                    ChartNames.emotional_response.value: bar_emotion_chart,
                    ChartNames.correctness.value: bar_correctness_chart,
                    ChartNames.experiential_matching.value: bar_experiential_chart,
                    ChartNames.conflict_information.value: bar_conflict_chart,
                    ChartNames.problem_importance.value: bar_problem_importance_chart,
                }
            )
            msv_graphs.append(parts)
            msv_bar_graphs.append(bar_parts)
            if system_number == 1:
                global selected_nodes
                plot, selected_nodes = create_system_two_node_graph(
                    system_two_state[id]
                )
                system_two_graph_components = components(plot)
    return templates.TemplateResponse(
        request=request,
        name="msv_visualizer.html",
        context={
            "msv_graphs": msv_graphs,
            "msv_json": msv_response,
            "msv_bar_graphs": msv_bar_graphs,
            "system_two_graph": system_two_graph_components,
        },
    )


@app.get("/node/{node_id}", response_class=HTMLResponse)
async def node_detail(node_id: int):
    """HTMX endpoint for node details"""
    info = selected_nodes[node_id]

    return f"""
    <div class="node-detail">
        <p><strong>Role:</strong> {info.node_role.replace("_", " ").title()}</p>
        <p><strong>Response:</strong> {info.node_response}</p>
        <p><em>Node ID: {node_id}</em></p>
    </div>
    """


excluded_keys = {"calculated_value", "version"}


def _clean_values(value) -> dict[str, int]:
    return {
        k: v
        for k, v in asdict(value).items()
        if k not in excluded_keys and not k.startswith("weight_")
    }


def _generate_bar_chart(data: dict[str, int], x_label: str, chart_title: str) -> figure:
    categories: list[str] = [
        k.replace("_", " ")
        .title()
        .replace(" ", "\x00", 1)
        .replace(" ", "\n", 1)
        .replace("\x00", " ")
        for k in data.keys()
    ]
    values: list[float] = list(data.values())
    p = figure(
        x_range=categories,
        # All vectors are on the 0-100 interval
        y_range=(0, 100),
        title=chart_title,
        toolbar_location=None,
        tools="",
    )
    p.vbar(x=categories, top=values, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = "Values"
    return p


def _generate_chart(data: dict[str, int], x_label: str, chart_title: str) -> figure:
    categories: list[str] = [
        k.replace("_", " ")
        .title()
        .replace(" ", "\x00", 1)
        .replace(" ", "\n", 1)
        .replace("\x00", " ")
        for k in data.keys()
    ]
    # categories: list[str] = list([k.replace("_", " ").title() for k in data.keys()])
    values: list[float] = list(data.values())

    num_vars = len(categories)

    # Calculate angles for each axis (in radians)
    angles = [i * 2 * math.pi / num_vars for i in range(num_vars)]
    angles.append(angles[0])  # Close the polygon

    # Close the values list to complete the polygon
    closed_values = values + [values[0]]

    # Convert polar coordinates to cartesian
    max_val = 100  # Your data is on 0-100 scale
    x = [val * math.cos(angle) for val, angle in zip(closed_values, angles)]
    y = [val * math.sin(angle) for val, angle in zip(closed_values, angles)]

    # Create axis lines from center to perimeter
    axis_x = [[0, max_val * math.cos(angle)] for angle in angles[:-1]]
    axis_y = [[0, max_val * math.sin(angle)] for angle in angles[:-1]]

    # Position labels outside the chart
    label_distance = max_val * 1.15
    label_x = [label_distance * math.cos(angle) for angle in angles[:-1]]
    label_y = [label_distance * math.sin(angle) for angle in angles[:-1]]

    # Create data sources
    polygon_source = ColumnDataSource(data=dict(x=x, y=y))
    axis_source = ColumnDataSource(data=dict(xs=axis_x, ys=axis_y))
    label_source = ColumnDataSource(data=dict(x=label_x, y=label_y, text=categories))

    # Create figure
    p = figure(
        width=500,
        height=500,
        title=chart_title,
        toolbar_location=None,
        tools="",
        match_aspect=True,
        x_range=(-max_val * 1.7, max_val * 1.7),
        y_range=(-max_val * 1.7, max_val * 1.7),
    )

    # Hide axes and grid
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False

    # Draw axis lines (spokes)
    p.multi_line(xs="xs", ys="ys", source=axis_source, color="#cccccc", line_width=1)

    # Draw concentric circles for reference (optional)
    circle_radii = [25, 50, 75, 100]
    for radius in circle_radii:
        circle_angles = [i * 2 * math.pi / 100 for i in range(101)]
        circle_x = [radius * math.cos(a) for a in circle_angles]
        circle_y = [radius * math.sin(a) for a in circle_angles]
        p.line(circle_x, circle_y, color="#eeeeee", line_width=1, alpha=0.5)

    # Draw the data polygon
    p.patch(
        x="x",
        y="y",
        source=polygon_source,
        alpha=0.3,
        color="#3298dc",
        line_color="#2366d1",
        line_width=2,
    )

    # Add category labels
    p.text(
        x="x",
        y="y",
        text="text",
        source=label_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="10pt",
    )

    return p


@app.post("/system1")
async def run_experiment(
    request: Request, system_one_prompt: SystemOnePrompt
) -> SystemOneResponse:
    response, response_id = await run_system_one(system_one_prompt.user_input)
    return SystemOneResponse(response=response, response_id=response_id)


def save_msv_state(msv_system_one, msv_system_two: MetacognitiveVector | None) -> str:
    id = str(uuid4())
    msv_state[id].append(msv_system_one)
    if msv_system_two:
        msv_state[id].append(msv_system_two)
    return id


history = deque(maxlen=10)


async def run_system_one(user_input: str) -> tuple[str, str]:
    try:
        # Generate a response from the system one model and compute the metacognative state vector
        response = await system_one_model.get_response(user_input, list(history))
        historical_info = "\n".join(
            [
                message["content"]
                for message in history
                if message["role"] == "assistant"
            ]
        )
        state = await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
            compute_method=system_configuration.vector_computation_key,
            prompts=system_configuration.prompts,
            weights=system_configuration.weights,
            response=response,
            original_prompt=user_input,
            knowledge_base=historical_info,
            historical_responses=historical_info,
            additional_config=system_configuration.additional_configuration,
        )

        parsed_response = system_two_model.SystemTwoResponse(
            system_two_response=None, metacognitive_vector=None, node_responses=None
        )
        if MetacognitiveActivationComputation.should_engage_system_two(
            system_configuration.activation_computation_key,
            state,
            system_configuration.additional_configuration,
        ):
            parsed_response = await orchistrator.get_system_two_response(
                user_prompt=user_input,
                system_one_response=response,
                system_one_vector=state,
            )

        if session_id:
            record_interaction(
                db_file=f"data/{session_id}.sqlite3",
                user_prompt=user_input,
                system_one_response=response,
                system_one_msv=state,
                system_two_response=parsed_response.system_two_response,
                system_two_msv=parsed_response.metacognitive_vector,
            )
        id = save_msv_state(state, parsed_response.metacognitive_vector)
        system_two_state[id] = parsed_response
        history.append({"role": "user", "content": user_input})
        system_response = (
            (
                parsed_response.system_two_response
                if parsed_response.system_two_response
                else response
            ),
            id,
        )
        history.append({"role": "assistant", "content": system_response[0]})

        return system_response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "weights_and_prompts": system_configuration.model_dump()},
    )


session_id: str | None = None
system_configuration: SystemConfiguration = SystemConfiguration()


@app.post("/reset", response_class=HTMLResponse)
async def reset_system(configuration: dict[str, Any] | None = None) -> str:
    utc_now = datetime.now(timezone.utc)
    formatted_datetime = utc_now.strftime("%Y-%m-%d_%H_%M_%S_%f")
    data_directory = Path("data")
    data_directory.mkdir(parents=True, exist_ok=True)

    global system_configuration

    if configuration:
        system_configuration = SystemConfiguration.model_validate(configuration)
    else:
        system_configuration = SystemConfiguration()

    orchistrator.set_configuration(system_configuration)

    created = create_database_and_table(
        f"data/{formatted_datetime}.sqlite3", system_configuration.model_dump()
    )
    msv_state.clear()
    system_two_state.clear()
    history.clear()

    if created:
        global session_id
        session_id = formatted_datetime
    return f"""<!-- {session_id} -->
<div class="notification is-success">
    <button class="delete"></button>
    Configuration saved successfully!
</div>
<div id="chatbox" class="box" style="height: 400px; overflow-y: auto;" hx-swap-oob="true">
    <!-- Chat messages will be appended here -->
</div>
<input id="user_input" class="input" type="text" name="user_input" placeholder="Type your message..." required hx-swap-oob="true">
<div id="system-details" hx-swap-oob="true">
    <!-- System details will be loaded here -->
</div>
"""


if __name__ == "__main__":
    import uvicorn

    orchistrator = Orchestrator(system_configuration=system_configuration)

    uvicorn.run(app, host="0.0.0.0", port=8000)
