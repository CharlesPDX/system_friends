import ollama

async def get_response(message: str) -> str:
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content":message}])
    return response.message.content
