
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
class MetacognitiveVector:
    emotional_response: EmotionalResponse
    correctness: int
    experiential_matching: int
    conflict_information: int
    problem_importance: int

    def should_engage_system_two(self) -> bool:
        return False

async def compute_metacognitive_state_vector(message: str) -> MetacognitiveVector:
    emotional_response, correctness, experiential_matching, conflict_information, problem_importance = await asyncio.gather(_compute_emotional_response(message), 
                                                                                                                            _compute_correctness(message), 
                                                                                                                            _compute_experiential_matching(message), 
                                                                                                                            _compute_conflict_information(message), 
                                                                                                                            _compute_problem_importance(message))
    
    
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


async def _compute_correctness(message:str) -> int:
    return 0

async def _compute_experiential_matching(message:str) -> int:
    return 0

async def _compute_conflict_information(message:str) -> int:
    return 0

async def _compute_problem_importance(message:str) -> int:
    return 0
