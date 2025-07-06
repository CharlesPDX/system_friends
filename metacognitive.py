
import asyncio
from dataclasses import dataclass
import json

import ollama

@dataclass(kw_only=True)
class EmotionalResponse:
    emotion: str
    confidence: int

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
    content = f"""Categorize the text's emotional tone as either 'neutral' or identify the presence of one of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust). 
Also provide a numeric confidence score, and return it in JSON format {{"emotion": "emotion", "confidence": confidence}}, do not include any additional text
Text: {message}"""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content":content}])
    try:
        parsed_response = json.loads(response.message.content)
        return EmotionalResponse(emotion=parsed_response["emotion"],confidence=int(parsed_response["confidence"] * 100))
    except:
        return EmotionalResponse(emotion="unknown", confidence=0)

async def _compute_correctness(message:str) -> int:
    return 0

async def _compute_experiential_matching(message:str) -> int:
    return 0

async def _compute_conflict_information(message:str) -> int:
    return 0

async def _compute_problem_importance(message:str) -> int:
    return 0
