from dataclasses import dataclass

from simplexity.run_management.structured_configs import (
    GenerativeProcessConfig,
    LoggingConfig,
    MLFlowConfig,
    PredictiveModelConfig,
    OptimizerConfig,
    PersistenceConfig,
)


@dataclass
class Config:
    """Configuration for the managed run demo."""

    mlflow: MLFlowConfig
    logging: LoggingConfig
    generative_process: GenerativeProcessConfig
    persistence: PersistenceConfig
    predictive_model: PredictiveModelConfig
    optimizer: OptimizerConfig
    experiment_name: str
    run_name: str
    seed: int
    tags: dict[str, str]
