import ollama

from metacognitive import MetacognitiveVector
from prompts import PromptNames, Prompts

async def get_response(user_prompt: str, system_one_response:str, system_one_vector: MetacognitiveVector, prompts: Prompts) -> str:
    
    messages = [{"role": "system", "content": prompts.get_prompt(PromptNames.System_Two_System, context={}), "thinking": "true"},
                {"role": "assistant", "content":system_one_response},
                {"role": "user", "content":prompts.get_prompt(PromptNames.System_Two_User, context={"user_prompt": user_prompt})}]

    response = ollama.chat(model="llama3.2", messages=messages)
    return response.message.content
