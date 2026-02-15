import math
import statistics
from abc import abstractmethod
from typing import Self, TypeVar

from pydantic import BaseModel, computed_field

from common import NodeRole

T = TypeVar("T", bound=BaseModel)


class ResponseVectors(BaseModel):
    calculated_value: int = 0

    @abstractmethod
    def _compute_value(self) -> int: ...

    def model_post_init(self, context) -> None:
        self.calculated_value = self._compute_value()

    @staticmethod
    def mean(response_vectors: list[T], model: type[T]) -> T:
        self_fields = model.model_fields.keys()
        n = len(response_vectors)
        mean_model = {}
        for field in self_fields:
            if (
                field != "version"
                and field != "calculated_value"
                and not field.startswith("weight_")
            ):
                field_total = sum(
                    [getattr(response, field) for response in response_vectors]
                )
                mean_model[field] = field_total / n
        return model.model_validate(mean_model)


class EmotionalResponseWeights(BaseModel):
    weight_fear: float = 0.1
    weight_anger: float = 0.1
    weight_anticipation: float = 0.1
    weight_trust: float = 0.1
    weight_surprise: float = 0.1
    weight_positive: float = 0.1
    weight_negative: float = 0.1
    weight_sadness: float = 0.1
    weight_disgust: float = 0.1
    weight_joy: float = 0.1


class EmotionalResponse(ResponseVectors):
    version: str = "0.1"

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

    # weights: EmotionalResponseWeights
    weight_fear: float = 0.1
    weight_anger: float = 0.1
    weight_anticipation: float = 0.1
    weight_trust: float = 0.1
    weight_surprise: float = 0.1
    weight_positive: float = 0.1
    weight_negative: float = 0.1
    weight_sadness: float = 0.1
    weight_disgust: float = 0.1
    weight_joy: float = 0.1

    def _compute_value(self) -> int:
        self_fields = EmotionalResponse.model_fields.keys()
        running_total = 0.0
        for field in self_fields:
            if (
                field != "version"
                and field != "calculated_value"
                and not field.startswith("weight_")
            ):
                running_total += getattr(self, field) * getattr(self, f"weight_{field}")
        return min(int(running_total), 100)


class CorrectnessEvaluationResponse(ResponseVectors):
    version: str = "0.1"

    # Sum of weights should be 1
    logical_consistency: float
    weight_logical_consistency: float = 0.3

    factual_accuracy: float
    weight_factual_accuracy: float = 0.4

    contextual_appropriateness: float
    weight_contextual_appropriateness: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.logical_consistency * self.weight_logical_consistency)
                + (self.factual_accuracy * self.weight_factual_accuracy)
                + (
                    self.contextual_appropriateness
                    * self.weight_contextual_appropriateness
                )
            ),
            100,
        )


class ExperientialMatchingResponse(ResponseVectors):
    version: str = "0.11"

    # Weights are adaptive based on context. Should there be constraints on weights?
    knowledge_base_matching: float
    weight_knowledge_base_matching: float = 0.3

    historical_responses_matching: float
    weight_historical_responses_matching: float = 0.3

    cue_familiarity: float
    weight_cue_familiarity: float = 0.4

    def _compute_value(self) -> int:
        # This calculation assumes the matching values are in the range of [0,100]
        return min(
            int(
                (self.knowledge_base_matching * self.weight_knowledge_base_matching)
                + (
                    self.historical_responses_matching
                    * self.weight_historical_responses_matching
                )
            ),
            100,
        )


class ConflictingInformationResponse(ResponseVectors):
    version: str = "0.1"

    # Sum of weights should be 1
    internal_consistency: float
    weight_internal_consistency: float = 0.3

    source_agreement: float
    weight_source_agreement: float = 0.4

    temporal_stability: float
    weight_temporal_stability: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.internal_consistency * self.weight_internal_consistency)
                + (self.source_agreement * self.weight_source_agreement)
                + (self.temporal_stability * self.weight_temporal_stability)
            ),
            100,
        )


class ProblemImportanceResponse(ResponseVectors):
    version: str = "0.1"

    potential_consequences: float
    weight_potential_consequences: float = 0.4

    temporal_urgency: float
    weight_temporal_urgency: float = 0.3

    scope_of_impact: float
    weight_scope_of_impact: float = 0.3

    def _compute_value(self) -> int:
        return min(
            int(
                (self.potential_consequences * self.weight_potential_consequences)
                + (self.temporal_urgency * self.weight_temporal_urgency)
                + (self.scope_of_impact * self.weight_scope_of_impact)
            ),
            100,
        )


class MetacognitiveVector(ResponseVectors):
    version: str = "0.12"
    emotional_response: EmotionalResponse
    weight_emotional_response: float = 0.2

    correctness_evaluation: CorrectnessEvaluationResponse
    weight_correctness_evaluation: float = 0.2

    experiential_matching: ExperientialMatchingResponse
    weight_experiential_matching: float = 0.2

    conflicting_information: ConflictingInformationResponse
    weight_conflicting_information: float = 0.2

    problem_importance: ProblemImportanceResponse
    weight_problem_importance: float = 0.2

    def __hash__(self) -> int:
        return id(self)

    @computed_field
    @property
    def uncertainty(self) -> int:
        return max(100 - self.correctness_evaluation.calculated_value, 0)

    @computed_field
    @property
    def novelty(self) -> int:
        return max(100 - self.experiential_matching.calculated_value, 0)

    def _compute_value(self) -> int:
        return int(
            (self.emotional_response._compute_value() * self.weight_emotional_response)
            + (
                self.correctness_evaluation._compute_value()
                * self.weight_correctness_evaluation
            )
            + (
                self.experiential_matching._compute_value()
                * self.weight_experiential_matching
            )
            + (
                self.conflicting_information._compute_value()
                * self.weight_conflicting_information
            )
            + (
                self.problem_importance._compute_value()
                * self.weight_problem_importance
            )
        )

    @staticmethod
    def msv_mean(
        metacognitive_vectors: list["MetacognitiveVector"],
    ) -> "MetacognitiveVector":
        return MetacognitiveVector(
            emotional_response=EmotionalResponse.mean(
                [msv.emotional_response for msv in metacognitive_vectors],
                EmotionalResponse,
            ),
            correctness_evaluation=CorrectnessEvaluationResponse.mean(
                [msv.correctness_evaluation for msv in metacognitive_vectors],
                CorrectnessEvaluationResponse,
            ),
            experiential_matching=ExperientialMatchingResponse.mean(
                [msv.experiential_matching for msv in metacognitive_vectors],
                ExperientialMatchingResponse,
            ),
            conflicting_information=ConflictingInformationResponse.mean(
                [msv.conflicting_information for msv in metacognitive_vectors],
                ConflictingInformationResponse,
            ),
            problem_importance=ProblemImportanceResponse.mean(
                [msv.problem_importance for msv in metacognitive_vectors],
                ProblemImportanceResponse,
            ),
        )


class MetacognitiveActivationComputation:
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
    def should_engage_system_two(
        cls,
        compute_method: str,
        msv_by_role: dict[NodeRole, MetacognitiveVector],
        additional_config: dict = {},
    ) -> bool:
        if compute_method not in cls._registry:
            raise KeyError(f"No subclass registered with key '{compute_method}'")
        if compute_method not in cls._instances:
            cls._instances[compute_method] = cls._registry[compute_method]()
        return cls._instances[compute_method]._should_engage_system_two(
            msv_by_role, additional_config
        )

    @classmethod
    def get_activation_result(
        cls,
        compute_method: str,
        metacognitive_vector: MetacognitiveVector,
        additional_config: dict = {},
    ) -> str:
        if compute_method not in cls._registry:
            raise KeyError(f"No subclass registered with key '{compute_method}'")
        if compute_method not in cls._instances:
            cls._instances[compute_method] = cls._registry[compute_method]()
        return cls._instances[compute_method]._get_activation_result(
            metacognitive_vector, additional_config
        )

    @abstractmethod
    def _should_engage_system_two(
        self,
        msv_by_role: dict[NodeRole, MetacognitiveVector],
        additional_config: dict = {},
    ) -> bool: ...

    @abstractmethod
    def _get_activation_result(
        self,
        metacognitive_vector: MetacognitiveVector,
        additional_config: dict = {},
    ) -> str: ...


class BaselineMetacognitiveActivationComputation(MetacognitiveActivationComputation):
    compute_method = "baseline"
    activation_threshold: float = 0.1

    def _mean_msv(self, metacognitive_vectors: list[MetacognitiveVector]) -> int:
        return int(
            statistics.mean([msv.calculated_value for msv in metacognitive_vectors])
        )

    def _should_engage_system_two(
        self,
        msv_by_role: dict[NodeRole, MetacognitiveVector],
        additional_config: dict = {},
    ) -> bool:
        activation_value = self._activation_function(
            self._mean_msv(list(msv_by_role.values()))
        )
        return activation_value >= additional_config.get(
            "activation_threshold", self.activation_threshold
        )

    def _get_activation_result(
        self,
        metacognitive_vector: MetacognitiveVector,
        additional_config: dict = {},
    ) -> str:
        return str(self._activation_function(metacognitive_vector.calculated_value))

    def _activation_function(self, value: int) -> float:
        return 1 / (1 + math.exp(-value * 0.00001))


def generate_empty_msv() -> MetacognitiveVector:
    emotional_response = EmotionalResponse(
        fear=0,
        anger=0,
        anticipation=0,
        trust=0,
        surprise=0,
        positive=0,
        negative=0,
        sadness=0,
        disgust=0,
        joy=0,
        # weights=EmotionalResponseWeights(),
    )
    correctness = CorrectnessEvaluationResponse(
        logical_consistency=0.0, factual_accuracy=0.0, contextual_appropriateness=0.0
    )
    experiential_matching = ExperientialMatchingResponse(
        knowledge_base_matching=0.0,
        historical_responses_matching=0.0,
        cue_familiarity=0.0,
    )
    conflicting_information = ConflictingInformationResponse(
        internal_consistency=0.0, source_agreement=0.0, temporal_stability=0.0
    )
    problem_importance = ProblemImportanceResponse(
        potential_consequences=0.0, temporal_urgency=0.0, scope_of_impact=0.0
    )

    return MetacognitiveVector(
        emotional_response=emotional_response,
        correctness_evaluation=correctness,
        experiential_matching=experiential_matching,
        conflicting_information=conflicting_information,
        problem_importance=problem_importance,
    )
