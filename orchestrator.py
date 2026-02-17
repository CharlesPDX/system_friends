import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from common import MetacognitiveComponentNames
from config import SystemConfiguration
from metacognitive import (
    MetacognitiveActivationComputation,
    MetacognitiveVector,
    RoutingResult,
)
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


@dataclass
class AssignmentResult:
    """Complete result of role assignment, including both matrices."""

    # Heterogeneous matrix: each row from agent's own preliminary MSV
    heterogeneous_matrix: np.ndarray | None
    heterogeneous_scores: list[dict[NodeRole, float]] | None

    # Hungarian assignment (from heterogeneous matrix, or degenerate if no prelim)
    assignment: dict[str, int]  # role_name -> agent_index
    assignment_total_fitness: float

    # The per-agent preliminary MSVs (None if no preliminary responses generated)
    agent_msvs: list[MetacognitiveVector] | None

    # Which model each agent used (None if no preliminary responses)
    agent_model_names: list[str] | None

    # Fitness mode used
    fitness_mode: str  # "code_weights" or "paper_equations"


class SystemResponse(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    system_one_response: str
    system_two_response: str | None
    final_response: str
    metacognitive_vector: MetacognitiveVectorResponse
    node_responses: list[NodeResponse] | None

    routing: RoutingResult

    # Role assignment (only populated if System 2 is engaged)
    assignment: AssignmentResult | None

    # Generalist annotation (only if MSV triggers fire)
    generalist_annotation: str | None


class Orchestrator:
    # TODO adjust weights!
    role_weights: dict[NodeRole, dict[MetacognitiveComponentNames, float]] = {
        NodeRole.Domain_Expert: {
            MetacognitiveComponentNames.Emotional_Response: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation: 0.7,
            MetacognitiveComponentNames.Experiential_Matching: 0.0,
            MetacognitiveComponentNames.Conflicting_Information: 0.1,
            MetacognitiveComponentNames.Problem_Importance: 0.2,
        },
        NodeRole.Critic: {
            MetacognitiveComponentNames.Emotional_Response: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation: 0.5,
            MetacognitiveComponentNames.Experiential_Matching: 0.05,
            MetacognitiveComponentNames.Conflicting_Information: 0.4,
            MetacognitiveComponentNames.Problem_Importance: 0.05,
        },
        NodeRole.Evaluator: {
            MetacognitiveComponentNames.Emotional_Response: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation: 0.4,
            MetacognitiveComponentNames.Experiential_Matching: 0.0,
            MetacognitiveComponentNames.Conflicting_Information: 0.3,
            MetacognitiveComponentNames.Problem_Importance: 0.3,
        },
        NodeRole.Synthesizer: {
            MetacognitiveComponentNames.Emotional_Response: 0.0,
            MetacognitiveComponentNames.Correctness_Evaluation: 0.25,
            MetacognitiveComponentNames.Experiential_Matching: 0.25,
            MetacognitiveComponentNames.Conflicting_Information: 0.25,
            MetacognitiveComponentNames.Problem_Importance: 0.25,
        },
        NodeRole.Generalist: {
            MetacognitiveComponentNames.Emotional_Response: 0.2,
            MetacognitiveComponentNames.Correctness_Evaluation: 0.2,
            MetacognitiveComponentNames.Experiential_Matching: 0.2,
            MetacognitiveComponentNames.Conflicting_Information: 0.2,
            MetacognitiveComponentNames.Problem_Importance: 0.2,
        },
    }

    def _get_role_preferences_core(
        self,
        msv: MetacognitiveVector,
        role_weights: dict[NodeRole, dict[MetacognitiveComponentNames, float]],
    ) -> dict[NodeRole, float]:
        role_preferences: dict[NodeRole, float] = {}
        for role, weights in role_weights.items():
            running_value = 0.0
            for vector_name, weight in weights.items():
                # maybe find a better way to do this than a really flexi-typed
                # accessor into ResponseVectors
                if vector_name == MetacognitiveComponentNames.Uncertainty:
                    running_value += weight * msv.uncertainty
                    continue
                running_value += weight * getattr(msv, vector_name).calculated_value
            role_preferences[role] = running_value
        return role_preferences

    def _get_role_preferences(self, msv: MetacognitiveVector) -> dict[NodeRole, float]:
        return self._get_role_preferences_core(msv, self.role_weights)

    def _get_alternate_role_preference(
        self, msv: MetacognitiveVector
    ) -> dict[NodeRole, float]:
        # --- PAPER WEIGHTS (IJCAI Equations 1-5) ---
        # These implement the equations from the IJCAI demo paper.
        # IMPORTANT: The specific weight VALUES (alpha_i, beta_i, etc.) are NOT
        # specified in the paper — only the DIMENSIONS each role uses.
        # The values below are initial calibration values that need empirical
        # validation.
        #
        # Paper Eq 1:  f_Expert(m)  = α₁·CE + α₂·EM
        # Paper Eq 2:  f_Critic(m)  = β₁·CI + β₂·(1-CE)     ← NOTE: (1-CE) inversion
        # Paper Eq 3:  f_Eval(m)    = γ₁·CE + γ₂·CI + γ₃·EM
        # Paper Eq 4:  f_Synth(m)   = δ₁·EM + δ₂·CE
        # Paper Eq 5:  f_Gen(m)     = ε₁·PI + ε₂·CI + ε₃·ER
        #

        role_based_specific_weights: dict[
            NodeRole, dict[MetacognitiveComponentNames, float]
        ] = {
            NodeRole.Domain_Expert: {
                MetacognitiveComponentNames.Correctness_Evaluation: 0.6,
                MetacognitiveComponentNames.Correctness_Evaluation: 0.4,
            },
            NodeRole.Critic: {
                MetacognitiveComponentNames.Conflicting_Information: 0.5,
                MetacognitiveComponentNames.Uncertainty: 0.5,
            },  # inv_ce = (1 - CE)
            NodeRole.Evaluator: {
                MetacognitiveComponentNames.Correctness_Evaluation: 0.35,
                MetacognitiveComponentNames.Conflicting_Information: 0.35,
                MetacognitiveComponentNames.Correctness_Evaluation: 0.30,
            },
            NodeRole.Synthesizer: {
                MetacognitiveComponentNames.Correctness_Evaluation: 0.55,
                MetacognitiveComponentNames.Correctness_Evaluation: 0.45,
            },
            NodeRole.Generalist: {
                MetacognitiveComponentNames.Problem_Importance: 0.40,
                MetacognitiveComponentNames.Conflicting_Information: 0.35,
                MetacognitiveComponentNames.Emotional_Response: 0.25,
            },
        }
        return self._get_role_preferences_core(msv, role_based_specific_weights)

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
        if Path("nodes.json").exists():
            with open("nodes.json") as nodes_config_file:
                nodes_config = NodesConfig.model_validate_json(nodes_config_file.read())

        pipeline_length = len(self.role_weights)

        if nodes_config and nodes_config.nodes:
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
            self._nodes = [Node() for _ in enumerate(self.role_weights.keys())]

        self.assigned_roles: defaultdict[NodeRole, list[Node]] = defaultdict(list)
        number_of_available_nodes = len(self._nodes)
        for role_index, role in enumerate(self.role_weights.keys()):
            self.assigned_roles[role].append(
                self._nodes[role_index % number_of_available_nodes]
            )

    def _reset_assigned_roles(self) -> None:
        for role in self.assigned_roles.keys():
            self.assigned_roles[role].clear()

    def _transition_nodes(
        self,
        msv_by_role: dict[NodeRole, list[MetacognitiveVector]],
    ) -> AssignmentResult:
        self._reset_assigned_roles()
        fitness_scores = [
            # self._get_role_preferences(msv) TODO for this branch!
            self._get_alternate_role_preference(msv)
            for msvs in msv_by_role.values()
            for msv in msvs
        ]
        n_agents = len(fitness_scores)
        m_roles = len(self.assigned_roles)

        cost_matrix = np.zeros((n_agents, m_roles))
        for i, scores in enumerate(fitness_scores):
            for j, role in enumerate(self.assigned_roles.keys()):
                cost_matrix[i, j] = scores[role]

        # Handle rectangular matrices (more agents than roles or vice versa)
        # TODO - issue here if roles != nodes, either unassigned roles or
        # unassigned nodes
        row_indices: Iterable[int]
        col_indices: Iterable[int]
        row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)

        assignment: dict[str, int] = {}
        total_fitness = 0.0
        all_roles = list(self.assigned_roles.keys())
        for row, col in zip(row_indices, col_indices):
            if col < m_roles:  # safety check
                role = all_roles[col]
                assignment[role.value] = row
                total_fitness += fitness_scores[row][role]
                self.assigned_roles[role].append(self._nodes[row])
        return AssignmentResult(
            heterogeneous_matrix=np.array(
                [[fs[role] for role in all_roles] for fs in fitness_scores]
            ),
            heterogeneous_scores=fitness_scores,
            assignment=assignment,
            assignment_total_fitness=total_fitness,
            agent_msvs=[msv for msvs in msv_by_role.values() for msv in msvs],
            agent_model_names=[node.model for node in self._nodes],
            fitness_mode="paper_equations",
        )

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
        assignment = None
        generalist_annotation = None
        system_one_response = await self._nodes[0].summarize_responses(
            list([value[0] for value in system_one_responses.values()])
        )
        msv_by_role: defaultdict[NodeRole, list[MetacognitiveVector]] = defaultdict(
            list
        )
        for msv, role in awaited_msvs:
            msv_by_role[role].append(msv[0])

        engage_system_two = MetacognitiveActivationComputation.should_engage_system_two(
            self.system_configuration.activation_computation_key,
            msv_by_role,
            self.system_configuration.additional_configuration,
        )
        if engage_system_two:
            (
                node_responses,
                overall_system_two_response,
                system_two_msv,
                assignment,
                generalist_annotation,
            ) = await self._get_system_two_response(
                user_prompt=user_prompt,
                system_one_response=system_one_response,
                msv_by_role=msv_by_role,
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
            routing=MetacognitiveActivationComputation.get_routing_result(
                self.system_configuration.activation_computation_key,
                msv_by_role,
                self.system_configuration.additional_configuration,
            ),
            generalist_annotation=generalist_annotation,
            assignment=assignment,
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

        assignment = self._transition_nodes(msv_by_role)

        node_responses: list[NodeResponse] = []
        previous_response = system_one_response
        previous_role = NodeRole.System_One
        previous_state: list[MetacognitiveVector] | None = None

        for role, nodes in self.assigned_roles.items():
            if role == NodeRole.Generalist:
                continue

            stage_node_responses = await asyncio.gather(
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
            response_by_node = {
                node: response for response, node in stage_node_responses
            }

            all_states = await asyncio.gather(
                *[
                    MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                        self.system_configuration,
                        node,
                        node_response,
                        previous_response,
                    )
                    for node_response, node in stage_node_responses
                ]
            )
            node_responses.extend(
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
                stage_node_responses[0][0]
                if len(stage_node_responses) == 1
                else await nodes[0].summarize_responses(
                    [response[0] for response in stage_node_responses]
                )
            )
            previous_role = role
            previous_state = [state for state, _ in all_states]
            messages.append({"role": "assistant", "content": previous_response})

            # basic early stopping implementation
            # mean_state = MetacognitiveVector.msv_mean(
            #     list([msv for msv in previous_state])
            # )
            # if (
            #     mean_state.correctness_evaluation.calculated_value >= 85
            #     or mean_state.conflicting_information.calculated_value <= 20
            # ):
            #     break

        last_mean_msv = (
            MetacognitiveVector.msv_mean(list([msv for msv in previous_state]))
            if previous_state
            else None
        )
        generalist_annotation = None
        if last_mean_msv is not None and (
            last_mean_msv.conflicting_information.calculated_value > 65
            or last_mean_msv.emotional_response.calculated_value < 35
            or last_mean_msv.problem_importance.calculated_value > 70
        ):
            generalist_annotation, generalist_node = await self.assigned_roles[
                NodeRole.Generalist
            ][0].get_response(
                user_prompt,
                previous_response,
                previous_role,
                self.system_configuration.prompts,
                NodeRole.Generalist,
            )
            generalist_msv, _ = (
                await MetacognitiveVectorComputation.compute_metacognitive_state_vector(
                    self.system_configuration,
                    generalist_node,
                    generalist_annotation,
                    previous_response,
                )
            )

            node_responses.append(
                NodeResponse(
                    node_role=NodeRole.Generalist,
                    node_response=generalist_annotation,
                    node_msv=generalist_msv,
                )
            )

        overall_system_two_response = previous_response
        state = last_mean_msv

        return (
            node_responses,
            overall_system_two_response,
            state,
            assignment,
            generalist_annotation,
        )
