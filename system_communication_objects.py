from pydantic import BaseModel

from metacognitive import MetacognitiveVector

class SystemTwoRequest(BaseModel):
    user_prompt: str
    system_one_response: str
    metacognitive_vector: MetacognitiveVector


class SystemTwoResponse(BaseModel):
    system_two_response: str
    metacognitive_vector: MetacognitiveVector
