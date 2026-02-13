import json
from dataclasses import asdict
from typing import Self

from pydantic import BaseModel, Field, model_validator

from metacognitive import MetacognitiveVector, generate_empty_msv
from prompts import Prompts


class SystemConfiguration(BaseModel):
    prompts: Prompts = Field(default_factory=Prompts)
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    vector_computation_key: str = Field(default="baseline")
    activation_computation_key: str = Field(default="baseline")
    additional_configuration: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    def _set_additional_configuration(cls, values: dict) -> dict:
        if "additional_configuration" in values:
            try:
                values["additional_configuration"] = json.loads(
                    values["additional_configuration"]
                )
            except Exception:
                print("could not load additional configuration")
                values["additional_configuration"] = {}
        return values

    @model_validator(mode="after")
    def _default_weights(self) -> Self:
        if not self.weights:
            self.weights = self._get_weights(generate_empty_msv())
        return self

    def _get_weights(self, msv: MetacognitiveVector) -> dict[str, dict[str, float]]:
        weights = {}
        for x in (
            ("msv_weights", msv),
            ("emotional_response", msv.emotional_response),
            ("correctness", msv.correctness),
            ("experiential_matching", msv.experiential_matching),
            ("conflict_information", msv.conflict_information),
            ("problem_importance", msv.problem_importance),
        ):
            weights[x[0]] = {
                k: v for k, v in asdict(x[1]).items() if k.startswith("weight")
            }
        return weights
