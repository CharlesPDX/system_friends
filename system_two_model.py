import ollama

from metacognitive import MetacognitiveVector

async def get_response(user_prompt: str, system_one_response:str, system_one_vector: MetacognitiveVector) -> str:
    
    messages = [{"role": "system", "content":"You are a System Two, logical analytical deep thinking system", "thinking": "true"},
                {"role": "assistant", "content":system_one_response},
                {"role": "user", "content":f"""Given the previous System One response, and its interpretation of the user's orginal prompt: '{user_prompt}', what would you say instead?"""}]

    response = ollama.chat(model="llama3.2", messages=messages)
    return response.message.content
