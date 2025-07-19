import asyncio
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from metacognitive import compute_metacognitive_state_vector
import system_one_model


app = FastAPI()

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        # Generate a response from the system one model and compute the metacognative state vector
        response = await system_one_model.get_response(user_input)
        state = await compute_metacognitive_state_vector(response, user_input)
        print(state)

        return f'<div class="message"><div class="message-body">Bot: {response}</div></div>'

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
