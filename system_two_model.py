from enum import StrEnum, auto

import ollama
from pydantic import BaseModel

from metacognitive import MetacognitiveVector
from prompts import PromptNames, Prompts
from system_communication_objects import SystemTwoRequest


class NodeRole(StrEnum):
    Domain_Expert = auto()
    Critic = auto()
    Evaluator = auto()
    Generalist = auto()
    Synthesizer = auto()
    System_One = auto()


class NodeResponse(BaseModel):
    node_role: str
    node_response: str
    node_msv: MetacognitiveVector

    class Config:
        frozen = True


class SystemTwoResponse(BaseModel):
    system_two_response: str | None
    metacognitive_vector: MetacognitiveVector | None
    node_responses: list[NodeResponse] | None


class Node:
    def __init__(self, role: NodeRole = NodeRole.Generalist) -> None:
        self.role = role

    role: NodeRole = NodeRole.Generalist
    # TODO adjust weights!
    role_weights: dict[NodeRole, dict[str, float]] = {
        NodeRole.Domain_Expert: {
            "emotional_response": 0.0,
            "correctness_evaluation": 0.7,
            "experiential_matching": 0.0,
            "conflicting_information": 0.1,
            "problem_importance": 0.2,
        },
        NodeRole.Critic: {
            "emotional_response": 0.0,
            "correctness_evaluation": 0.5,
            "experiential_matching": 0.05,
            "conflicting_information": 0.4,
            "problem_importance": 0.05,
        },
        NodeRole.Evaluator: {
            "emotional_response": 0.0,
            "correctness_evaluation": 0.4,
            "experiential_matching": 0.0,
            "conflicting_information": 0.3,
            "problem_importance": 0.3,
        },
        NodeRole.Generalist: {
            "emotional_response": 0.2,
            "correctness_evaluation": 0.2,
            "experiential_matching": 0.2,
            "conflicting_information": 0.2,
            "problem_importance": 0.2,
        },
        NodeRole.Synthesizer: {
            "emotional_response": 0.0,
            "correctness_evaluation": 0.25,
            "experiential_matching": 0.25,
            "conflicting_information": 0.25,
            "problem_importance": 0.25,
        },
    }

    def get_role_preferences(
        self, system_one_vector: MetacognitiveVector
    ) -> dict[NodeRole, float]:
        role_preferences: dict[NodeRole, float] = {}
        for role, weights in self.role_weights.items():
            running_value = 0.0
            for vector_name, weight in weights.items():
                # maybe find a better way to do this than a really flexi-typed accessor into ResponseVectors
                running_value += (
                    weight * getattr(system_one_vector, vector_name).calculated_value
                )
            role_preferences[role] = running_value
        return role_preferences

    def assign_role(self, new_role: NodeRole) -> None:
        # TODO? keep role history?
        # TODO? update role weights?
        self.role = new_role

    def get_response(
        self,
        user_prompt: str,
        previous_node_response: str,
        previous_node_role: NodeRole,
        prompts: Prompts,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": prompts.get_prompt(
                    PromptNames(f"{self.role}_system"),
                    context={"previous_node_role": previous_node_role},
                ),
                "thinking": "true",
            },
            {"role": "assistant", "content": previous_node_response},
            {
                "role": "user",
                "content": prompts.get_prompt(
                    PromptNames(f"{self.role}_user"),
                    context={"user_prompt": user_prompt},
                ),
            },
        ]

        response = ollama.chat(model="llama3.2", messages=messages)
        return response.message.content
