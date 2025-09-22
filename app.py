import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone


from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

from experiment_model import SystemOnePrompt, SystemOneResponse
from history import create_database_and_table, record_interaction
from metacognitive import compute_metacognitive_state_vector
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
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    response = await run_system_one(user_input)
    return f'<div class="message"><div class="message-body">Bot: {response}</div></div>'


@app.post("/system1")
async def run_experiment(request: Request, system_one_prompt: SystemOnePrompt) -> SystemOneResponse:
    response = await run_system_one(system_one_prompt.user_input)
    return SystemOneResponse(response=response, session_id=session_id)


async def run_system_one(user_input:str):
    try:
        # Generate a response from the system one model and compute the metacognative state vector
        response = await system_one_model.get_response(user_input)
        state = await compute_metacognitive_state_vector(response, user_input)
        
        # print(state.calculated_value)
        # print(state.compute_value())
        parsed_response = SystemTwoResponse(system_two_response=None, metacognitive_vector=None)
        if state.should_engage_system_two():
            system_two_response = httpx.post(f"{app_args.system_two_url}/system2", 
                                             content=SystemTwoRequest(user_prompt=user_input, 
                                                                      system_one_response=response, 
                                                                      metacognitive_vector=state).model_dump_json(), 
                                                                      timeout=None)
            parsed_response = SystemTwoResponse.model_validate_json(system_two_response.text)

        if session_id:
            record_interaction(db_file=f"{session_id}.sqlite3", 
                                user_prompt=user_input,
                                system_one_response=response,
                                system_one_msv=state,
                                system_two_response=parsed_response.system_two_response,
                                system_two_msv=parsed_response.metacognitive_vector)
        return parsed_response.system_two_response if parsed_response.system_two_response else response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system2")
async def run_system_two(system_two_request: SystemTwoRequest) -> SystemTwoResponse:
    # This requires running a second instance with the `--system-two`` flag:
    response = await system_two_model.get_response(system_two_request.user_prompt, 
                                                   system_two_request.system_one_response, 
                                                   system_two_request.metacognitive_vector)
    state = await compute_metacognitive_state_vector(response, system_two_request.system_one_response)

    return SystemTwoResponse(system_two_response=response, metacognitive_vector=state)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

session_id: str | None = None

@app.post("/reset")
async def reset_system() -> None:
    utc_now = datetime.now(timezone.utc)
    formatted_datetime = utc_now.strftime("%Y-%m-%d_%H_%M_%S_%f")
    created = create_database_and_table(f"{formatted_datetime}.sqlite3")
    
    if created:
        global session_id 
        session_id = formatted_datetime



if __name__ == "__main__":
    import uvicorn
    port = "8000" if not app_args.system_two else "8001"
    uvicorn.run(app, host="0.0.0.0", port=port)
