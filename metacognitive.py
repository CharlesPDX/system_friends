
from abc import abstractmethod
import asyncio
from dataclasses import dataclass
import json

from nrclex import NRCLex
import ollama

@dataclass
class ResponseVectors:
    @abstractmethod
    def compute_value(self) -> int: ...

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
class ProblemImportance(ResponseVectors):
    potential_consequences: float
    weight_potential_consequences: float = 0.3

    temporal_urgency: float
    weight_temporal_urgency: float = 0.3

    scope_of_impact: float
    weight_scope_of_impact: float = 0.3

    def compute_value(self) -> int:
        return min(int(
                (self.potential_consequences * self.weight_potential_consequences) + 
                (self.temporal_urgency * self.weight_temporal_urgency) + 
                (self.scope_of_impact * self.weight_scope_of_impact)), 100)

@dataclass(kw_only=True)
class MetacognitiveVector:
    emotional_response: EmotionalResponse
    correctness: CorrectnessResponse
    experiential_matching: int
    conflict_information: int
    problem_importance: ProblemImportance

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
    # remove vestigial(?) "anticip" in favor of the populated "anticipation",
    # seems like sometimes "anticipation" is populated sometimes "anticip" ?
    if "anticipation" not in text_object.affect_frequencies:
        if "anticip" in text_object.affect_frequencies:
            text_object.affect_frequencies["anticipation"] = text_object.affect_frequencies["anticip"]
            del text_object.affect_frequencies["anticip"]
        else:
            text_object.affect_frequencies["anticipation"] = 0.0
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

async def _compute_problem_importance(original_prompt:str) -> ProblemImportance:
    content = f"""How would you assess the User Prompt for problem importance on the dimensions of potential consequences, temporal urgency, and scope of impact? 
Assess each dimension from 0 to 100 and return the response in JSON format {{"potential_consequences": "potential consequences", "temporal_urgency": "temporal urgency", "scope_of_impact": "scope of impact"}}, 
do not include any additional text.
User Prompt: {original_prompt}"""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content":content}])
    try:
        parsed_response = json.loads(response.message.content)
        return ProblemImportance(potential_consequences=float(parsed_response["potential_consequences"]), temporal_urgency=float(parsed_response["temporal_urgency"]), scope_of_impact=float(parsed_response["scope_of_impact"]))
    except:
        return ProblemImportance(potential_consequences=0.0, temporal_urgency=0.0, scope_of_impact=0.0)
