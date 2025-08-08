from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

from metacognitive import compute_metacognitive_state_vector
from system_communication_objects import SystemTwoRequest, SystemTwoResponse
import system_one_model
import system_two_model


app = FastAPI()

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    # this requires running an instance such as:
    # uvicorn app:app --reload --port 8000 
    try:
        # Generate a response from the system one model and compute the metacognative state vector
        response = await system_one_model.get_response(user_input)
        state = await compute_metacognitive_state_vector(response, user_input)
        
        # print(state.calculated_value)
        # print(state.compute_value())
        if state.should_engage_system_two():
            # TODO make host configurable
            system_two_response = httpx.post("http://127.0.0.1:8001/system2", content=SystemTwoRequest(user_prompt=user_input, system_one_response=response, metacognitive_vector=state).model_dump_json(), timeout=None)
            parsed_response = SystemTwoResponse.model_validate_json(system_two_response.text)
            return f'<div class="message"><div class="message-body">Bot: {parsed_response.system_two_response}</div></div>'

        return f'<div class="message"><div class="message-body">Bot: {response}</div></div>'

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system2")
async def run_system_two(system_two_request: SystemTwoRequest) -> SystemTwoResponse:
    # This requires running a second instance such as:
    # uvicorn app:app --reload --port 8001
    response = await system_two_model.get_response(system_two_request.user_prompt, system_two_request.system_one_response, system_two_request.metacognitive_vector)
    state = await compute_metacognitive_state_vector(response, system_two_request.system_one_response)

    return SystemTwoResponse(system_two_response=response, metacognitive_vector=state)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0")
