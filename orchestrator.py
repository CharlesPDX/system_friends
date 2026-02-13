import ollama

from config import SystemConfiguration
from metacognitive import MetacognitiveVector, MetacognitiveVectorComputation
from prompts import PromptNames
from system_two_model import Node, NodeResponse, NodeRole, SystemTwoResponse


class Orchestrator:
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

    async def get_system_two_response(
        self,
        user_prompt: str,
        system_one_response: str,
        system_one_vector: MetacognitiveVector,
    ) -> SystemTwoResponse:

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

        return SystemTwoResponse(
            node_responses=role_responses,
            system_two_response=overall_system_two_response,
            metacognitive_vector=state,
        )
