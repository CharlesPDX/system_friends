
import asyncio
from dataclasses import dataclass
import json

from nrclex import NRCLex
import ollama

@dataclass(kw_only=True)
class EmotionalResponse:
    fear: float
    anger: float
    anticipation: float
    trust: float
    surprise: float
    positive: float
    negative: float
    sadness: float
    disgust: float
    joy: float

@dataclass(kw_only=True)
class CorrectnessResponse:
    logical_consistency: float
    factual_accuracy: float
    contextual_appropriateness: float

@dataclass(kw_only=True)
class MetacognitiveVector:
    emotional_response: EmotionalResponse
    correctness: CorrectnessResponse
    experiential_matching: int
    conflict_information: int
    problem_importance: int

    def should_engage_system_two(self) -> bool:
        return False

async def compute_metacognitive_state_vector(response: str, original_prompt: str) -> MetacognitiveVector:
    emotional_response, correctness, experiential_matching, conflict_information, problem_importance = await asyncio.gather(_compute_emotional_response(response), 
                                                                                                                            _compute_correctness(response, original_prompt), 
                                                                                                                            _compute_experiential_matching(response), 
                                                                                                                            _compute_conflict_information(response), 
                                                                                                                            _compute_problem_importance(response))
    
    
    return MetacognitiveVector(emotional_response=emotional_response, 
                               correctness=correctness, 
                               experiential_matching=experiential_matching, 
                               conflict_information=conflict_information, 
                               problem_importance=problem_importance)

async def _compute_emotional_response(message: str) -> EmotionalResponse:
    text_object = NRCLex(message)
    # remove vestigial(?) "anticip" in favor of the populated "anticipation"
    del text_object.affect_frequencies["anticip"]
    return EmotionalResponse(**text_object.affect_frequencies)


async def _compute_correctness(message:str, original_prompt: str) -> CorrectnessResponse:
    content = f"""Without citing modern fact-checks, how would you assess this claim on the dimensions of logical consistency, factual accuracy, and contextual appropriateness? 
Consider the contextual appropriateness with the given context. 
Assess each dimension from 0 to 100 and return the response in JSON format {{"logical_consistency": "logical consistency", "factual_accuracy": "factual accuracy", "contextual_appropriateness": "contextual appropriateness"}}, 
do not include any additional text.
Context: {original_prompt}
Claim: {message}"""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content":content}])
    try:
        parsed_response = json.loads(response.message.content)
        return CorrectnessResponse(logical_consistency=parsed_response["logical_consistency"] ,factual_accuracy=int(parsed_response["factual_accuracy"]), contextual_appropriateness=int(parsed_response["contextual_appropriateness"]))
    except:
        return CorrectnessResponse(logical_consistency=0.0, factual_accuracy=0.0, contextual_appropriateness=0.0)


async def _compute_experiential_matching(message:str) -> int:
    return 0

async def _compute_conflict_information(message:str) -> int:
    return 0

async def _compute_problem_importance(message:str) -> int:
    return 0
