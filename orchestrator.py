import asyncio
from collections import deque

from pydantic import BaseModel

from common import MetacognitiveComponentNames
from config import SystemConfiguration
from metacognitive import MetacognitiveActivationComputation, MetacognitiveVector
from prompts import PromptNames
from system_nodes import MetacognitiveVectorComputation, Node, NodeResponse, NodeRole


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
    # TODO adjust weights!
    role_weights: dict[NodeRole, dict[str, float]] = {
        NodeRole.Domain_Expert: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.7,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.0,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.1,
            MetacognitiveComponentNames.Problem_Importance.value: 0.2,
        },
        NodeRole.Critic: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.5,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.05,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.4,
            MetacognitiveComponentNames.Problem_Importance.value: 0.05,
        },
        NodeRole.Evaluator: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.4,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.0,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.3,
            MetacognitiveComponentNames.Problem_Importance.value: 0.3,
        },
        NodeRole.Generalist: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.2,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.2,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.2,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.2,
            MetacognitiveComponentNames.Problem_Importance.value: 0.2,
        },
        NodeRole.Synthesizer: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.25,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.25,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.25,
            MetacognitiveComponentNames.Problem_Importance.value: 0.25,
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

    history = deque(maxlen=10)

    def __init__(self, system_configuration: SystemConfiguration):
        self.system_configuration = system_configuration
        self._nodes = [
            Node(),
            Node(),
        ]
        self.assigned_roles: dict[NodeRole, Node] = {
            NodeRole.Domain_Expert: self._nodes[0],
            NodeRole.Critic: self._nodes[1],
            NodeRole.Evaluator: self._nodes[0],
            NodeRole.Synthesizer: self._nodes[1],
        }

    def _reset_taken_nodes(self) -> None:
        self.assigned_roles: dict[NodeRole, Node] = {
            NodeRole.Domain_Expert: self._nodes[0],
            NodeRole.Critic: self._nodes[1],
            NodeRole.Evaluator: self._nodes[0],
            NodeRole.Synthesizer: self._nodes[1],
        }

    def _transition_nodes(
        self,
        msv_by_role: dict[NodeRole, MetacognitiveVector],
    ):
        self._reset_taken_nodes()
        # for

    def set_configuration(self, system_configuration: SystemConfiguration) -> None:
        self.system_configuration = system_configuration

    def reset(self) -> None:
        self.history.clear()

    def _get_historical_info_from_chat(self) -> str:
        return "\n".join(
            [
                message["content"]
                for message in self.history
                if message["role"] == "assistant"
            ]
        )

    async def _compute_metacognitive_state_vector(
        self,
        role: NodeRole,
        response: str,
        user_prompt: str,
        historical_info: str = "",
    ) -> tuple[MetacognitiveVector, NodeRole]:
        return (
            await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                system_configuration=self.system_configuration,
                node=self.assigned_roles[role],
                response=response,
                original_prompt=user_prompt,
                knowledge_base=historical_info,
                historical_responses=historical_info,
            ),
            role,
        )

    async def get_metacognitive_informed_response(
        self, user_prompt: str
    ) -> SystemResponse:
        system_one_responses: dict[NodeRole, str] = {}
        # Generate a response from the system one model and compute the metacognative state vector
        awaited_responses = await asyncio.gather(
            *[
                node.get_system_one_response(user_prompt, self.history, role)
                for role, node in self.assigned_roles.items()
            ]
        )
        system_one_responses = {role: response for response, role in awaited_responses}

        historical_info = self._get_historical_info_from_chat()
        awaited_msvs = await asyncio.gather(
            *[
                self._compute_metacognitive_state_vector(
                    node_role, response, user_prompt, historical_info
                )
                for node_role, response in system_one_responses.items()
            ]
        )
        overall_system_two_response = None
        system_two_msv = None
        node_responses = None
        system_one_response = await self._nodes[0].summarize_system_one_response(
            list(system_one_responses.values())
        )
        msv_by_role = {role: msv for msv, role in awaited_msvs}
        if MetacognitiveActivationComputation.should_engage_system_two(
            self.system_configuration.activation_computation_key,
            msv_by_role,
            self.system_configuration.additional_configuration,
        ):
            node_responses, overall_system_two_response, system_two_msv = (
                await self._get_system_two_response(
                    user_prompt=user_prompt,
                    system_one_response=system_one_response,
                    msv_by_role=msv_by_role,
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
                system_one_metacognitive_vector=MetacognitiveVector.msv_mean(
                    list(msv_by_role.values())
                ),
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
        msv_by_role: dict[NodeRole, MetacognitiveVector],
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

        self._transition_nodes(msv_by_role)

        role_responses: list[NodeResponse] = []
        previous_response = system_one_response
        previous_role = NodeRole.System_One
        synthesizer_response: str | None = None
        synthesizer_msv: MetacognitiveVector | None = None
        for role, node in self.assigned_roles.items():
            if node:
                node_response = await node.get_response(
                    user_prompt,
                    previous_response,
                    previous_role,
                    self.system_configuration.prompts,
                    role,
                )

                state = await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                    self.system_configuration,
                    node,
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

            overall_system_two_response = await self._nodes[0].get_response(
                user_prompt,
                previous_response,
                previous_role,
                self.system_configuration.prompts,
                NodeRole.Generalist,
            )
            state = (
                await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                    self.system_configuration,
                    self._nodes[0],
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
