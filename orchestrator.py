import asyncio
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from common import MetacognitiveComponentNames
from config import SystemConfiguration
from metacognitive import MetacognitiveActivationComputation, MetacognitiveVector
from prompts import PromptNames
from system_nodes import MetacognitiveVectorComputation, Node, NodeResponse, NodeRole


class NodeConfig(BaseModel):
    host: str
    port: str = Field(default="11434")
    model: str


class NodesConfig(BaseModel):
    nodes: list[NodeConfig]


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
        NodeRole.Synthesizer: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.25,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.25,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.25,
            MetacognitiveComponentNames.Problem_Importance.value: 0.25,
        },
        NodeRole.Generalist: {
            MetacognitiveComponentNames.Emotional_Response.value: 0.2,
            MetacognitiveComponentNames.Correctness_Evaluation.value: 0.2,
            MetacognitiveComponentNames.Experiential_Matching.value: 0.2,
            MetacognitiveComponentNames.Conflicting_Information.value: 0.2,
            MetacognitiveComponentNames.Problem_Importance.value: 0.2,
        },
    }

    def _get_role_preferences(self, msv: MetacognitiveVector) -> dict[NodeRole, float]:
        role_preferences: dict[NodeRole, float] = {}
        for role, weights in self.role_weights.items():
            running_value = 0.0
            for vector_name, weight in weights.items():
                # maybe find a better way to do this than a really flexi-typed
                # accessor into ResponseVectors
                running_value += weight * getattr(msv, vector_name).calculated_value
            role_preferences[role] = running_value
        return role_preferences

    history = deque(maxlen=10)
    _dialectic_pipeline = [
        NodeRole.Domain_Expert,
        NodeRole.Critic,
        NodeRole.Evaluator,
        NodeRole.Synthesizer,
    ]

    def __init__(self, system_configuration: SystemConfiguration):
        self.system_configuration = system_configuration
        nodes_config = None
        if Path("nodes.json").exists:
            with open("nodes.json") as nodes_config_file:
                nodes_config = NodesConfig.model_validate_json(nodes_config_file.read())

        pipeline_length = len(self._dialectic_pipeline)

        if nodes_config:
            self._nodes = [
                Node(
                    host=node_config.host,
                    port=node_config.port,
                    model=node_config.model,
                )
                for node_config in nodes_config.nodes
            ]

            config_nodes_index_limit = len(nodes_config.nodes) - 1
            config_node_index = 0
            while len(self._nodes) < pipeline_length:
                self._nodes.append(
                    Node(
                        host=nodes_config.nodes[config_node_index].host,
                        port=nodes_config.nodes[config_node_index].port,
                        model=nodes_config.nodes[config_node_index].model,
                    )
                )
                config_node_index += 1
                config_node_index %= config_nodes_index_limit

        else:
            self._nodes = [
                Node(),
            ] * pipeline_length

        self.assigned_roles: defaultdict[NodeRole, list[Node]] = defaultdict(list)
        number_of_available_nodes = len(self._nodes)
        for role_index, role in enumerate(self._dialectic_pipeline):
            self.assigned_roles[role].append(
                self._nodes[role_index % number_of_available_nodes]
            )

    def _reset_assigned_roles(self) -> None:
        for role in self._dialectic_pipeline:
            self.assigned_roles[role].clear()

    def _transition_nodes(
        self,
        msv_by_role: dict[NodeRole, list[MetacognitiveVector]],
    ):
        self._reset_assigned_roles()
        fitness_scores = [
            self._get_role_preferences(msv)
            for msvs in msv_by_role.values()
            for msv in msvs
        ]
        n_agents = len(fitness_scores)
        m_roles = len(self._dialectic_pipeline)

        cost_matrix = np.zeros((n_agents, m_roles))
        for i, scores in enumerate(fitness_scores):
            for j, role in enumerate(self._dialectic_pipeline):
                cost_matrix[i, j] = scores[role]

        # Handle rectangular matrices (more agents than roles or vice versa)
        # TODO - issue here if roles != nodes, either unassigned roles or
        # unassigned nodes
        row_indices: Iterable[int]
        col_indices: Iterable[int]
        row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)

        assignment: dict[NodeRole, int] = {}
        total_fitness = 0.0
        for row, col in zip(row_indices, col_indices):
            if col < m_roles:  # safety check
                role = self._dialectic_pipeline[col]
                assignment[role] = row
                total_fitness += fitness_scores[row][role]
                self.assigned_roles[role].append(self._nodes[row])
        print()

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
        node: Node,
        role: NodeRole,
        response: str,
        user_prompt: str,
        historical_info: str = "",
    ) -> tuple[tuple[MetacognitiveVector, Node], NodeRole]:
        return (
            await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                system_configuration=self.system_configuration,
                node=node,
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
        system_one_responses: dict[Node, tuple[str, NodeRole]] = {}
        # Generate a response from the system one model and compute the
        # metacognative state vector
        awaited_responses = await asyncio.gather(
            *[
                node.get_system_one_response(user_prompt, self.history, role)
                for role, nodes in self.assigned_roles.items()
                for node in nodes
            ]
        )
        system_one_responses = {
            node: (response, role) for response, role, node in awaited_responses
        }

        historical_info = self._get_historical_info_from_chat()
        awaited_msvs = await asyncio.gather(
            *[
                self._compute_metacognitive_state_vector(
                    node, value[1], value[0], user_prompt, historical_info
                )
                for node, value in system_one_responses.items()
            ]
        )
        overall_system_two_response = None
        system_two_msv = None
        node_responses = None
        system_one_response = await self._nodes[0].summarize_responses(
            list([value[0] for value in system_one_responses.values()])
        )
        msv_by_role: defaultdict[NodeRole, list[MetacognitiveVector]] = defaultdict(
            list
        )
        for msv, role in awaited_msvs:
            msv_by_role[role].append(msv[0])

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
                    list([msv for msvs in msv_by_role.values() for msv in msvs])
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
        msv_by_role: dict[NodeRole, list[MetacognitiveVector]],
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
        previous_state: list[MetacognitiveVector] | None = None

        for role, nodes in self.assigned_roles.items():

            node_responses = await asyncio.gather(
                *[
                    node.get_response(
                        user_prompt,
                        previous_response,
                        previous_role,
                        self.system_configuration.prompts,
                        role,
                    )
                    for node in nodes
                ]
            )
            response_by_node = {node: response for response, node in node_responses}

            all_states = await asyncio.gather(
                *[
                    MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                        self.system_configuration,
                        node,
                        node_response,
                        previous_response,
                    )
                    for node_response, node in node_responses
                ]
            )
            role_responses.extend(
                [
                    NodeResponse(
                        node_role=role,
                        node_response=response_by_node[node],
                        node_msv=state,
                    )
                    for state, node in all_states
                ]
            )

            previous_response = (
                node_responses[0][0]
                if len(node_responses) == 1
                else await nodes[0].summarize_responses(
                    [response[0] for response in node_responses]
                )
            )
            previous_role = role
            messages.append({"role": "assistant", "content": previous_response})

        overall_system_two_response = previous_response
        state = (
            MetacognitiveVector.msv_mean(list([msv for msv in previous_state]))
            if previous_state
            else None
        )

        return (
            role_responses,
            overall_system_two_response,
            state,
        )
