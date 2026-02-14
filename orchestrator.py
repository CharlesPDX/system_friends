from collections import deque

import ollama
from pydantic import BaseModel

import system_one_model
from config import SystemConfiguration
from metacognitive import (
    MetacognitiveActivationComputation,
    MetacognitiveVector,
    MetacognitiveVectorComputation,
)
from prompts import PromptNames
from system_two_model import Node, NodeResponse, NodeRole


class MetacognitiveVectorResponse(BaseModel):
    system_one_metacognitive_vector: MetacognitiveVector
    system_two_metacognitive_vector: MetacognitiveVector | None


class SystemResponse(BaseModel):
    system_one_response: str
    system_two_response: str | None
    final_response: str
    metacognitive_vector: MetacognitiveVectorResponse
    node_responses: list[NodeResponse] | None


class Orchestrator:
    history = deque(maxlen=10)

    def __init__(self, system_configuration: SystemConfiguration):
        self.system_configuration = system_configuration
        self._system_one_nodes = [Node()]
        self._system_two_nodes = [
            Node(role=NodeRole.Domain_Expert),
            Node(role=NodeRole.Critic),
        ]
        self.taken_roles: dict[NodeRole, Node | None] = {
            NodeRole.Domain_Expert: None,
            NodeRole.Critic: None,
            NodeRole.Evaluator: None,
            NodeRole.Generalist: None,
            NodeRole.Synthesizer: None,
        }

    def _reset_taken_nodes(self) -> None:
        self.taken_roles: dict[NodeRole, Node | None] = {
            NodeRole.Domain_Expert: None,
            NodeRole.Critic: None,
            NodeRole.Evaluator: None,
            NodeRole.Generalist: None,
            NodeRole.Synthesizer: None,
        }

    def _transition_nodes(self, system_one_vector: MetacognitiveVector):
        self._reset_taken_nodes()

        for node in self._system_two_nodes:
            # TODO: manage role balance w/ Hungarian algo, right now use first available
            role_preferences = node.get_role_preferences(system_one_vector)
            sorted_role_preferences = [
                role
                for role, _ in sorted(
                    role_preferences.items(), key=lambda r: r[1], reverse=True
                )
            ]
            for role in sorted_role_preferences:
                if self.taken_roles[role] == None:
                    self.taken_roles[role] = node
                    node.assign_role(role)
                    break

    def set_configuration(self, system_configuration: SystemConfiguration) -> None:
        self.system_configuration = system_configuration

    def reset(self) -> None:
        self.history.clear()

    async def get_metacognitive_informed_response(
        self, user_prompt: str
    ) -> SystemResponse:
        # Generate a response from the system one model and compute the metacognative state vector
        system_one_response = await system_one_model.get_response(
            user_prompt, list(self.history)
        )
        historical_info = "\n".join(
            [
                message["content"]
                for message in self.history
                if message["role"] == "assistant"
            ]
        )
        state = await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
            compute_method=self.system_configuration.vector_computation_key,
            prompts=self.system_configuration.prompts,
            weights=self.system_configuration.weights,
            response=system_one_response,
            original_prompt=user_prompt,
            knowledge_base=historical_info,
            historical_responses=historical_info,
            additional_config=self.system_configuration.additional_configuration,
        )
        overall_system_two_response = None
        system_two_msv = None
        node_responses = None
        if MetacognitiveActivationComputation.should_engage_system_two(
            self.system_configuration.activation_computation_key,
            state,
            self.system_configuration.additional_configuration,
        ):
            node_responses, overall_system_two_response, system_two_msv = (
                await self._get_system_two_response(
                    user_prompt=user_prompt,
                    system_one_response=system_one_response,
                    system_one_vector=state,
                )
            )

        system_response = SystemResponse(
            system_one_response=system_one_response,
            system_two_response=overall_system_two_response,
            final_response=(
                overall_system_two_response
                if overall_system_two_response
                else system_one_response
            ),
            metacognitive_vector=MetacognitiveVectorResponse(
                system_one_metacognitive_vector=state,
                system_two_metacognitive_vector=system_two_msv,
            ),
            node_responses=node_responses,
        )

        self.history.append({"role": "user", "content": user_prompt})
        self.history.append(
            {"role": "assistant", "content": system_response.final_response}
        )

        return system_response

    async def _get_system_two_response(
        self,
        user_prompt: str,
        system_one_response: str,
        system_one_vector: MetacognitiveVector,
    ) -> tuple:

        messages = [
            {
                "role": "system",
                "content": self.system_configuration.prompts.get_prompt(
                    PromptNames.System_Two_System, context={}
                ),
                "thinking": "true",
            },
            {"role": "assistant", "content": system_one_response},
        ]

        self._transition_nodes(system_one_vector)

        role_responses: list[NodeResponse] = []
        previous_response = system_one_response
        previous_role = NodeRole.System_One
        synthesizer_response: str | None = None
        synthesizer_msv: MetacognitiveVector | None = None
        for role, node in self.taken_roles.items():
            if node:
                node_response = node.get_response(
                    user_prompt,
                    previous_response,
                    previous_role,
                    self.system_configuration.prompts,
                )

                state = await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                    self.system_configuration.vector_computation_key,
                    self.system_configuration.prompts,
                    self.system_configuration.weights,
                    node_response,
                    previous_response,
                )
                role_responses.append(
                    NodeResponse(
                        node_role=role, node_response=node_response, node_msv=state
                    )
                )
                if role == NodeRole.Synthesizer:
                    synthesizer_response = node_response
                    synthesizer_msv = state

                previous_response = node_response
                previous_role = role
                messages.append({"role": "assistant", "content": node_response})
        if not synthesizer_response:
            messages.append(
                {
                    "role": "user",
                    "content": self.system_configuration.prompts.get_prompt(
                        PromptNames.System_Two_User,
                        context={"user_prompt": user_prompt},
                    ),
                }
            )

            overall_system_two_response = ollama.chat(
                model="llama3.2", messages=messages
            ).message.content
            state = (
                await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                    self.system_configuration.vector_computation_key,
                    self.system_configuration.prompts,
                    self.system_configuration.weights,
                    overall_system_two_response if overall_system_two_response else "",
                    system_one_response,
                )
            )
        else:
            overall_system_two_response = synthesizer_response
            state = synthesizer_msv

        return (
            role_responses,
            overall_system_two_response,
            state,
        )
