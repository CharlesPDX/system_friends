
from abc import abstractmethod
import asyncio
from dataclasses import dataclass, fields
import json
import math

from nrclex import NRCLex
import ollama

@dataclass
class ResponseVectors:
    @abstractmethod
    def compute_value(self) -> int: ...

@dataclass(kw_only=True)
class EmotionalResponse(ResponseVectors):
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

    def compute_value(self) -> int:
        self_fields = fields(self)
        equal_weight = 1 / len(self_fields)
        running_total = 0.0
        for field in self_fields:
            running_total += getattr(self, field.name) * 100 * equal_weight
        return min(int(running_total), 100)

@dataclass(kw_only=True)
class CorrectnessResponse(ResponseVectors):
    logical_consistency: float
    weight_logical_consitency: float = 0.3

    factual_accuracy: float
    weight_factual_accuracy: float = 0.3

    contextual_appropriateness: float
    weight_contextual_appropriateness:float = 0.3

    def compute_value(self) -> int:
        return min(int(
            (self.logical_consistency * self.weight_logical_consitency) + 
            (self.factual_accuracy * self.weight_factual_accuracy) + 
            (self.contextual_appropriateness * self.weight_contextual_appropriateness)), 100)

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
class MetacognitiveVector(ResponseVectors):
    emotional_response: EmotionalResponse
    weight_emotional_response: float = 0.2

    correctness: CorrectnessResponse
    weight_correctness: float = 0.2

    experiential_matching: int
    weight_experiential_matching: float = 0.2

    conflict_information: int
    weight_conflict_information: float = 0.2

    problem_importance: ProblemImportance
    weight_problem_importance: float = 0.2

    _activation_threshold: float = 0.2

    def should_engage_system_two(self) -> bool:
        activation_value = self._activation_function(self.compute_value())
        print(activation_value)
        return activation_value >= self._activation_threshold

    
    def _activation_function(self, value: int) -> float:
        return 1 / (1 + math.exp(-value))

    def compute_value(self) -> int:
        return int(
                    (self.emotional_response.compute_value() * self.weight_emotional_response) + 
                    (self.correctness.compute_value() * self.weight_correctness) +
                    (self.experiential_matching * self.weight_experiential_matching) +
                    (self.conflict_information * self.weight_conflict_information) +
                    (self.problem_importance.compute_value() * self.weight_problem_importance)
                )

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
        else:
            text_object.affect_frequencies["anticipation"] = 0.0
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
