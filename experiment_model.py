from pydantic import BaseModel, Field

from config import SystemConfiguration


class Experiment(BaseModel):
    id: str
    prompts: list[str]
    configuration: SystemConfiguration | None = Field(default=None)


class Experiments(BaseModel):
    experiments: list[Experiment]


class CompletedExperiment(BaseModel):
    experiment_id: str
    session_id: str
    errors: list[str]
    experiment_start: str
    duration_seconds: int


class CompletedExperiments(BaseModel):
    completed_experiments: list[CompletedExperiment] = Field(default=[])


class SystemOnePrompt(BaseModel):
    user_input: str


class SystemOneResponse(BaseModel):
    response: str
    response_id: str
