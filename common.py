from enum import StrEnum, auto


class MetacognitiveComponentNames(StrEnum):
    Emotional_Response = "emotional_response"
    Correctness_Evaluation = "correctness_evaluation"
    Experiential_Matching = "experiential_matching"
    Conflicting_Information = "conflicting_information"
    Problem_Importance = "problem_importance"


class NodeRole(StrEnum):
    Domain_Expert = auto()
    Critic = auto()
    Evaluator = auto()
    Generalist = auto()
    Synthesizer = auto()
    System_One = auto()
