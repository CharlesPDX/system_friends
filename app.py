import argparse
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from enum import StrEnum
import json
from uuid import uuid4
from pathlib import Path


from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
from bokeh.plotting import figure
from bokeh.embed import components

from experiment_model import SystemOnePrompt, SystemOneResponse
from history import create_database_and_table, record_interaction
from metacognitive import MetacognitiveVector, compute_metacognitive_state_vector
from prompts import Prompts
from system_communication_objects import SystemTwoRequest, SystemTwoResponse
import system_one_model
import system_two_model

parser = argparse.ArgumentParser()
parser.add_argument("--system-two", default=False, action="store_true")
parser.add_argument("--system-two-url", required=False)
app_args = parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not app_args.system_two:
        await reset_system()
    
    yield
    # cleanup/shutdown goes here, if necessary

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    response, id = await run_system_one(user_input)
    return f'''
<div class="message is-bot" 
     hx-get="/get_chart?id={id}" 
     hx-trigger="click" 
     hx-target="#system-details">
        <div class="message-body">Bot: {response}</div>
</div>'''

msv_state: defaultdict[str,list[MetacognitiveVector]] = defaultdict(list)

class ChartNames(StrEnum):
    overall_msv = "Overall MSV"
    emotional_response = "Emotional Response"
    correctness = "Correctness"
    experiential_matching = "Experiential Matching"
    conflict_information = "Conflict Information" 
    problem_importance = "Problem Importance"

excluded_keys = {"calculated_value", "version"}

@app.get("/get_chart", response_class=HTMLResponse)
async def get_chart(request: Request, id: str=None):
    msv_response = []
    msv_graphs = []
    
    if id:
        for system_number, msv in enumerate(msv_state.get(id, [])):
            system_label = f"System {system_number+1}"
            msv_response.append(json.dumps(asdict(msv) | {"activation_result": msv._activation_function(msv.calculated_value)}, indent=2))
            
            data = {"emotional_response": msv.emotional_response.calculated_value, 
                    "correctness": msv.correctness.calculated_value, 
                    "experiential_matching": msv.experiential_matching.calculated_value, 
                    "conflict_information": msv.conflict_information.calculated_value,
                    "problem_importance": msv.problem_importance.calculated_value, }
            emotional_data = _clean_values(msv.emotional_response)
            correctness_data =_clean_values(msv.correctness)
            experiential_matching_data = _clean_values(msv.experiential_matching)
            conflict_information_data = _clean_values(msv.conflict_information)
            problem_importance_data = _clean_values(msv.problem_importance)

            # Create a Bokeh plot
            msv_components_chart = _generate_chart(data, "MSV Components", f"{system_label} MSV")
            emotion_chart = _generate_chart(emotional_data, "Emotion Components", f"{system_label} Emotion Vector")
            correctness_chart = _generate_chart(correctness_data, "Correctness Components", f"{system_label} Correctness Vector")
            experiential_chart = _generate_chart(experiential_matching_data, "Experiential Components", f"{system_label} Experiential Vector")
            conflict_chart = _generate_chart(conflict_information_data, "Conflict Components", f"{system_label} Conflict Vector")
            problem_importance_chart = _generate_chart(problem_importance_data, "Problem Importance Components", f"{system_label} Problem Importance Vector")

            # Generate the plots' HTML
            parts = components({ChartNames.overall_msv.value: msv_components_chart, 
                                ChartNames.emotional_response.value: emotion_chart, 
                                ChartNames.correctness.value: correctness_chart,
                                ChartNames.experiential_matching.value: experiential_chart,
                                ChartNames.conflict_information.value: conflict_chart,
                                ChartNames.problem_importance.value: problem_importance_chart})
            msv_graphs.append(parts)

    return templates.TemplateResponse(
        request=request, name="msv_visualizer.html", context={"msv_graphs": msv_graphs, "msv_json": msv_response}
    )

def _clean_values(value) -> dict[str, float]:
    return {k:v for k, v in asdict(value).items() if k not in excluded_keys and not k.startswith("weight_")}

def _generate_chart(data: dict[str, float],
                    x_label: str,
                    chart_title: str) -> figure:
    categories: list[str] = list(data.keys())
    values: list[float] = list(data.values())
    p = figure(x_range=categories, 
               # All vectors are on the 0-100 interval
               y_range=(0, 100),
               title=chart_title,
               toolbar_location=None, 
               tools="")
    p.vbar(x=categories, top=values, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = "Values"
    return p
    

@app.post("/system1")
async def run_experiment(request: Request, system_one_prompt: SystemOnePrompt) -> SystemOneResponse:
    response = await run_system_one(system_one_prompt.user_input)
    return SystemOneResponse(response=response, session_id=session_id)

def save_msv_state(msv_system_one, msv_system_two: MetacognitiveVector) -> str:
    id = str(uuid4())
    msv_state[id].append(msv_system_one)
    if msv_system_two:
        msv_state[id].append(msv_system_two)
    return id

prompts = Prompts()

async def run_system_one(user_input:str) -> tuple[str, str]:
    try:
        global prompts
        # Generate a response from the system one model and compute the metacognative state vector
        response = await system_one_model.get_response(user_input)
        state = await compute_metacognitive_state_vector(prompts, response, user_input)
        
        parsed_response = SystemTwoResponse(system_two_response=None, metacognitive_vector=None)
        if state.should_engage_system_two():
            system_two_response = httpx.post(f"{app_args.system_two_url}/system2", 
                                             content=SystemTwoRequest(user_prompt=user_input, 
                                                                      system_one_response=response, 
                                                                      metacognitive_vector=state,
                                                                      prompts=prompts).model_dump_json(), 
                                                                      timeout=None)
            parsed_response = SystemTwoResponse.model_validate_json(system_two_response.text)

        if session_id:
            record_interaction(db_file=f"data/{session_id}.sqlite3", 
                                user_prompt=user_input,
                                system_one_response=response,
                                system_one_msv=state,
                                system_two_response=parsed_response.system_two_response,
                                system_two_msv=parsed_response.metacognitive_vector)
        id = save_msv_state(state, parsed_response.metacognitive_vector)
        
        return (parsed_response.system_two_response if parsed_response.system_two_response else response, id)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system2")
async def run_system_two(system_two_request: SystemTwoRequest) -> SystemTwoResponse:
    # This requires running a second instance with the `--system-two`` flag:
    global prompts
    response = await system_two_model.get_response(system_two_request.user_prompt, 
                                                   system_two_request.system_one_response, 
                                                   system_two_request.metacognitive_vector)
    state = await compute_metacognitive_state_vector(prompts, response, system_two_request.system_one_response)

    return SystemTwoResponse(system_two_response=response, metacognitive_vector=state)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

session_id: str | None = None

@app.post("/reset")
async def reset_system() -> None:
    # TODO take in prompts & weights
    utc_now = datetime.now(timezone.utc)
    formatted_datetime = utc_now.strftime("%Y-%m-%d_%H_%M_%S_%f")
    data_directory = Path("data")
    data_directory.mkdir(parents=True, exist_ok=True)
    created = create_database_and_table(f"data/{formatted_datetime}.sqlite3")
    msv_state.clear()
    
    if created:
        global session_id 
        session_id = formatted_datetime


if __name__ == "__main__":
    import uvicorn
    port = "8000" if not app_args.system_two else "8001"
    uvicorn.run(app, host="0.0.0.0", port=port)
