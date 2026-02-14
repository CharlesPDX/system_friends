import asyncio
import json
from abc import abstractmethod
from collections import deque
from typing import Self

import ollama
from nrclex import NRCLex
from pydantic import BaseModel

from common import MetacognitiveComponentNames, NodeRole
from config import SystemConfiguration
from metacognitive import (
    ConflictingInformationResponse,
    CorrectnessEvaluationResponse,
    EmotionalResponse,
    ExperientialMatchingResponse,
    MetacognitiveVector,
    ProblemImportanceResponse,
)
from prompts import PromptNames, Prompts


class NodeResponse(BaseModel):
    node_role: str
    node_response: str
    node_msv: MetacognitiveVector

    class Config:
        frozen = True


class Node:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "localhost",
        port: str = "11434",
    ) -> None:
        server_address = f"http://{host}:{port}"
        self.client = ollama.Client(host=server_address)
        self.model = model

    async def get_system_one_response(
        self,
        user_prompt: str,
        historical_messages: deque[dict],
        role: NodeRole,
    ) -> tuple[str, NodeRole]:
        response = self.client.chat(
            model=self.model,
            messages=list(historical_messages)
            + [{"role": "user", "content": user_prompt}],
        )
        return response.message.content, role

    async def summarize_system_one_response(self, responses: list[str]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize the following responses into a single response to the user: {'\n'.join(responses)}",
                }
            ],
        )
        return response.message.content

    async def get_response(
        self,
        user_prompt: str,
        previous_node_response: str,
        previous_node_role: NodeRole,
        prompts: Prompts,
        role: NodeRole,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": prompts.get_prompt(
                    PromptNames(f"{role}_system"),
                    context={"previous_node_role": previous_node_role},
                ),
                "thinking": "true",
            },
            {"role": "assistant", "content": previous_node_response},
            {
                "role": "user",
                "content": prompts.get_prompt(
                    PromptNames(f"{role}_user"),
                    context={"user_prompt": user_prompt},
                ),
            },
        ]

        response = self.client.chat(model=self.model, messages=messages)
        return response.message.content


class MetacognitiveVectorComputation:

    # Set by subclasses to register
    compute_method: str | None = None

    _registry: dict[str, type[Self]] = {}
    _instances: dict[str, Self] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.compute_method is None:
            raise ValueError("Expected subclass to set compute_method")
        cls._registry[cls.compute_method] = cls

    @classmethod
    async def compute_metacognitive_state_vector(
        cls,
        system_configuration: SystemConfiguration,
        node: Node,
        response: str,
        original_prompt: str,
        knowledge_base: str = "",
        historical_responses: str = "",
        sources: str = "",
        temporal_info: str = "",
    ) -> MetacognitiveVector:
        if system_configuration.vector_computation_key not in cls._registry:
            raise KeyError(
                f"No subclass registered with key '{system_configuration.vector_computation_key}'"
            )
        if system_configuration.vector_computation_key not in cls._instances:
            cls._instances[system_configuration.vector_computation_key] = cls._registry[
                system_configuration.vector_computation_key
            ]()

        return await cls._instances[
            system_configuration.vector_computation_key
        ]._compute_metacognitive_state_vector(
            system_configuration,
            node,
            response,
            original_prompt,
            knowledge_base,
            historical_responses,
            sources,
            temporal_info,
        )

    @abstractmethod
    async def _compute_metacognitive_state_vector(
        self,
        system_configuration: SystemConfiguration,
        node: Node,
        response: str,
        original_prompt: str,
        knowledge_base: str = "",
        historical_responses: str = "",
        sources: str = "",
        temporal_info: str = "",
    ) -> MetacognitiveVector: ...


class BaselineMetacognitiveVectorComputation(MetacognitiveVectorComputation):
    compute_method = "baseline"

    async def _compute_metacognitive_state_vector(
        self,
        system_configuration: SystemConfiguration,
        node: Node,
        response: str,
        original_prompt: str,
        knowledge_base: str = "",
        historical_responses: str = "",
        sources: str = "",
        temporal_info: str = "",
    ) -> MetacognitiveVector:
        prompts = system_configuration.prompts
        (
            emotional_response,
            correctness_evaluation,
            experiential_matching,
            conflicting_information,
            problem_importance,
        ) = await asyncio.gather(
            self._compute_emotional_response(
                response,
                system_configuration.weights[
                    MetacognitiveComponentNames.Emotional_Response.value
                ],
            ),
            self._compute_correctness(
                response,
                original_prompt,
                prompts,
                system_configuration.weights[
                    MetacognitiveComponentNames.Correctness_Evaluation.value
                ],
                node,
            ),
            self._compute_experiential_matching(
                response,
                knowledge_base,
                historical_responses,
                prompts,
                system_configuration.weights[
                    MetacognitiveComponentNames.Experiential_Matching.value
                ],
                node,
            ),
            self._compute_conflicting_information(
                response,
                sources,
                temporal_info,
                prompts,
                system_configuration.weights[
                    MetacognitiveComponentNames.Conflicting_Information.value
                ],
                node,
            ),
            self._compute_problem_importance(
                response,
                prompts,
                system_configuration.weights[
                    MetacognitiveComponentNames.Problem_Importance.value
                ],
                node,
            ),
        )

        return MetacognitiveVector(
            emotional_response=emotional_response,
            correctness_evaluation=correctness_evaluation,
            experiential_matching=experiential_matching,
            conflicting_information=conflicting_information,
            problem_importance=problem_importance,
            **system_configuration.weights["msv_weights"],
        )

    async def _compute_emotional_response(
        self, message: str, weights: dict[str, float]
    ) -> EmotionalResponse:
        text_object = NRCLex(message)
        # remove vestigial(?) "anticip" in favor of the populated "anticipation",
        # seems like sometimes "anticipation" is populated sometimes "anticip" ?
        if "anticipation" not in text_object.affect_frequencies:
            if "anticip" in text_object.affect_frequencies:
                text_object.affect_frequencies["anticipation"] = (
                    text_object.affect_frequencies["anticip"]
                )
            else:
                text_object.affect_frequencies["anticipation"] = 0.0
        del text_object.affect_frequencies["anticip"]
        return EmotionalResponse(
            **{k: v * 100 for k, v in text_object.affect_frequencies.items()}
        )

    async def _compute_correctness(
        self,
        message: str,
        original_prompt: str,
        prompts: Prompts,
        weights: dict[str, float],
        node: Node,
    ) -> CorrectnessEvaluationResponse:
        content = prompts.get_prompt(
            PromptNames.Correctness_Evaluation,
            {"original_prompt": original_prompt, "message": message},
        )
        response = node.client.chat(
            model=node.model, messages=[{"role": "user", "content": content}]
        )
        try:
            parsed_response = json.loads(response.message.content)
            return CorrectnessEvaluationResponse(
                logical_consistency=parsed_response["logical_consistency"],
                factual_accuracy=int(parsed_response["factual_accuracy"]),
                contextual_appropriateness=int(
                    parsed_response["contextual_appropriateness"]
                ),
                **weights,
            )
        except:
            return CorrectnessEvaluationResponse(
                logical_consistency=0.0,
                factual_accuracy=0.0,
                contextual_appropriateness=0.0,
                **weights,
            )

    # Depending how to input knowledge base and historical responses, the prompt template would be different.
    # How to prompt to get matching level? options: matching level [0,100], similarity [0,1]
    async def _compute_experiential_matching(
        self,
        message: str,
        knowledge_base: str,
        historical_responses: str,
        prompts: Prompts,
        weights: dict[str, float],
        node: Node,
    ) -> ExperientialMatchingResponse:
        content = prompts.get_prompt(
            PromptNames.Experiential_Matching,
            {
                "knowledge_base": knowledge_base,
                "message": message,
                "historical_responses": historical_responses,
            },
        )
        response = node.client.chat(
            model=node.model, messages=[{"role": "user", "content": content}]
        )
        try:
            parsed_response = json.loads(response.message.content)
            return ExperientialMatchingResponse(
                knowledge_base_matching=float(
                    parsed_response["knowledge_base_matching"]
                ),
                historical_responses_matching=float(
                    parsed_response["historical_responses_matching"]
                ),
                cue_familiarity=float(parsed_response["cue_familiarity"]),
                **weights,
            )
        except:
            return ExperientialMatchingResponse(
                knowledge_base_matching=0.0,
                historical_responses_matching=0.0,
                cue_familiarity=0.0,
                **weights,
            )

    async def _compute_conflicting_information(
        self,
        message: str,
        sources: str,
        temporal_info: str,
        prompts: Prompts,
        weights: dict[str, float],
        node: Node,
    ) -> ConflictingInformationResponse:
        content = prompts.get_prompt(
            PromptNames.Conflicting_Information,
            {"sources": sources, "message": message, "temporal_info": temporal_info},
        )
        response = node.client.chat(
            model=node.model, messages=[{"role": "user", "content": content}]
        )
        try:
            parsed_response = json.loads(response.message.content)
            return ConflictingInformationResponse(
                internal_consistency=float(
                    parsed_response.get("internal_consistency", 0.0)
                ),
                source_agreement=float(parsed_response.get("source_agreement", 0.0)),
                temporal_stability=float(
                    parsed_response.get("temporal_stability", 0.0)
                ),
                **weights,
            )
        except:
            return ConflictingInformationResponse(
                internal_consistency=0.0,
                source_agreement=0.0,
                temporal_stability=0.0,
                **weights,
            )

    async def _compute_problem_importance(
        self,
        original_prompt: str,
        prompts: Prompts,
        weights: dict[str, float],
        node: Node,
    ) -> ProblemImportanceResponse:
        content = prompts.get_prompt(
            PromptNames.Problem_Importance, {"original_prompt": original_prompt}
        )
        response = node.client.chat(
            model=node.model, messages=[{"role": "user", "content": content}]
        )
        try:
            parsed_response = json.loads(response.message.content)
            return ProblemImportanceResponse(
                potential_consequences=float(parsed_response["potential_consequences"]),
                temporal_urgency=float(parsed_response["temporal_urgency"]),
                scope_of_impact=float(parsed_response["scope_of_impact"]),
                **weights,
            )
        except:
            return ProblemImportanceResponse(
                potential_consequences=0.0,
                temporal_urgency=0.0,
                scope_of_impact=0.0,
                **weights,
            )
