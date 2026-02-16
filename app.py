import json
import math
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
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

import system_nodes
from common import MetacognitiveComponentNames
from config import SystemConfiguration
from experiment_model import SystemOnePrompt, SystemOneResponse
from history import create_database_and_table, record_interaction
from metacognitive import (
    MetacognitiveActivationComputation,
    MetacognitiveVector,
    generate_empty_msv,
)
from orchestrator import MetacognitiveVectorResponse, Orchestrator, SystemResponse
from role_visualization import create_role_pipeline_components


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
    response, id = await _run_system(user_input)
    return f"""
<div class="message is-bot" 
     hx-get="/get_chart?id={id}" 
     hx-trigger="click" 
     hx-target="#system-details">
        <div class="message-body">Bot: {response}</div>
</div>"""


msv_state: defaultdict[str, list[MetacognitiveVector]] = defaultdict(list)
system_state: dict[str, SystemResponse] = {}
selected_nodes: list[system_nodes.NodeResponse] = []


class ChartNames(StrEnum):
    overall_msv = "Overall MSV"
    emotional_response = "Emotional Response"
    correctness_evaluation = "Correctness Evaluation"
    experiential_matching = "Experiential Matching"
    conflicting_information = "Conflicting Information"
    problem_importance = "Problem Importance"


@app.get("/get_chart", response_class=HTMLResponse)
async def get_chart(request: Request, id: str | None = None):
    msv_response = []
    msv_graphs = []
    msv_bar_graphs = []
    viz_components = {}
    result = None

    if id:
        for system_number, msv in enumerate(msv_state.get(id, [])):
            system_label = f"System {system_number+1}"
            msv_response.append(
                json.dumps(
                    msv.model_dump()
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
                MetacognitiveComponentNames.Emotional_Response.value: msv.emotional_response.calculated_value,
                MetacognitiveComponentNames.Correctness_Evaluation.value: msv.correctness_evaluation.calculated_value,
                MetacognitiveComponentNames.Experiential_Matching.value: msv.experiential_matching.calculated_value,
                MetacognitiveComponentNames.Conflicting_Information.value: msv.conflicting_information.calculated_value,
                MetacognitiveComponentNames.Problem_Importance.value: msv.problem_importance.calculated_value,
            }
            emotional_data = _clean_values(msv.emotional_response)
            correctness_evaluation_data = _clean_values(msv.correctness_evaluation)
            experiential_matching_data = _clean_values(msv.experiential_matching)
            conflicting_information_data = _clean_values(msv.conflicting_information)
            problem_importance_data = _clean_values(msv.problem_importance)

            # Create a Bokeh plot
            msv_components_chart = _generate_chart(
                data, "MSV Components", f"{system_label} MSV"
            )
            emotion_chart = _generate_chart(
                emotional_data, "Emotion Components", f"{system_label} Emotion Vector"
            )
            correctness_chart = _generate_chart(
                correctness_evaluation_data,
                "Correctness Components",
                f"{system_label} Correctness Vector",
            )
            experiential_chart = _generate_chart(
                experiential_matching_data,
                "Experiential Components",
                f"{system_label} Experiential Vector",
            )
            conflict_chart = _generate_chart(
                conflicting_information_data,
                "Conflicting Components",
                f"{system_label} Conflicting Vector",
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
                correctness_evaluation_data,
                "Correctness Components",
                f"{system_label} Correctness Vector",
            )
            bar_experiential_chart = _generate_bar_chart(
                experiential_matching_data,
                "Experiential Components",
                f"{system_label} Experiential Vector",
            )
            bar_conflict_chart = _generate_bar_chart(
                conflicting_information_data,
                "Conflicting Components",
                f"{system_label} Conflicting Vector",
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
                    ChartNames.correctness_evaluation.value: correctness_chart,
                    ChartNames.experiential_matching.value: experiential_chart,
                    ChartNames.conflicting_information.value: conflict_chart,
                    ChartNames.problem_importance.value: problem_importance_chart,
                }
            )
            bar_parts = components(
                {
                    ChartNames.overall_msv.value: bar_msv_components_chart,
                    ChartNames.emotional_response.value: bar_emotion_chart,
                    ChartNames.correctness_evaluation.value: bar_correctness_chart,
                    ChartNames.experiential_matching.value: bar_experiential_chart,
                    ChartNames.conflicting_information.value: bar_conflict_chart,
                    ChartNames.problem_importance.value: bar_problem_importance_chart,
                }
            )
            msv_graphs.append(parts)
            msv_bar_graphs.append(bar_parts)

        result = system_state[id]
        global selected_nodes
        viz_components, selected_nodes = create_role_pipeline_components(
            role_responses=result.node_responses or [],
            assignment_result=result.assignment,
            routing_result=result.routing,
            generalist_annotation=None,  # result.generalist_annotation,
        )

    return templates.TemplateResponse(
        request=request,
        name="msv_visualizer.html",
        context={
            "viz": viz_components,
            "msv_graphs": msv_graphs,
            "msv_json": msv_response,
            "msv_bar_graphs": msv_bar_graphs,
        },
    )


@app.get("/node/{node_id}", response_class=HTMLResponse)
async def node_detail(node_id: int):
    """HTMX endpoint for node details"""
    info = selected_nodes[node_id]
    msv = info.node_msv

    return f"""
    <div class="node-detail">
        <p><strong>Role:</strong> {info.node_role.replace("_", " ").title()}</p>
        <p><strong>MSV:</strong>
            ER={msv.emotional_response.calculated_value},
            CE={msv.correctness_evaluation.calculated_value},
            EM={msv.experiential_matching.calculated_value},
            CI={msv.conflicting_information.calculated_value},
            PI={msv.problem_importance.calculated_value}
        </p>
        <p><strong>Response:</strong> {info.node_response}</p>
    </div>
    """


excluded_keys = {"calculated_value", "version"}


def _clean_values(value) -> dict[str, int]:
    return {
        k: v
        for k, v in value.model_dump().items()
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


@app.post("/run_system")
async def run_system(
    request: Request, system_one_prompt: SystemOnePrompt
) -> SystemOneResponse:
    response, response_id = await _run_system(system_one_prompt.user_input)
    return SystemOneResponse(response=response, response_id=response_id)


def _save_msv_state(msv_response: MetacognitiveVectorResponse) -> str:
    id = str(uuid4())
    msv_state[id].append(msv_response.system_one_metacognitive_vector)
    if msv_response.system_two_metacognitive_vector:
        msv_state[id].append(msv_response.system_two_metacognitive_vector)
    return id


async def _run_system(user_input: str) -> tuple[str, str]:
    try:
        system_response = await orchestrator.get_metacognitive_informed_response(
            user_input
        )
        if session_id:
            record_interaction(
                db_file=f"data/{session_id}.sqlite3",
                user_prompt=user_input,
                system_one_response=system_response.system_one_response,
                system_one_msv=system_response.metacognitive_vector.system_one_metacognitive_vector,
                system_two_response=system_response.system_two_response,
                system_two_msv=system_response.metacognitive_vector.system_two_metacognitive_vector,
            )

        id = _save_msv_state(system_response.metacognitive_vector)
        system_state[id] = system_response

        return system_response.final_response, id
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "weights_and_prompts": system_configuration.model_dump()},
    )


def _get_default_weights() -> dict[str, dict[str, float]]:
    msv = generate_empty_msv()
    weights = {}
    for x in (
        ("msv_weights", msv),
        ("emotional_response", msv.emotional_response),
        ("correctness_evaluation", msv.correctness_evaluation),
        ("experiential_matching", msv.experiential_matching),
        ("conflicting_information", msv.conflicting_information),
        ("problem_importance", msv.problem_importance),
    ):
        weights[x[0]] = {
            k: v for k, v in x[1].model_dump().items() if k.startswith("weight")
        }
    return weights


session_id: str | None = None
system_configuration: SystemConfiguration = SystemConfiguration(
    weights=_get_default_weights()
)


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
        system_configuration = SystemConfiguration(weights=_get_default_weights())

    orchestrator.set_configuration(system_configuration)
    orchestrator.reset()

    created = create_database_and_table(
        f"data/{formatted_datetime}.sqlite3", system_configuration.model_dump()
    )
    msv_state.clear()
    system_state.clear()

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

    orchestrator = Orchestrator(system_configuration=system_configuration)

    uvicorn.run(app, host="0.0.0.0", port=8000)
