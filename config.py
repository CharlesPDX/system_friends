import json

from pydantic import BaseModel, Field, model_validator

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
