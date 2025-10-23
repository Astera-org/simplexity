from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as GenerativeProcessConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.mlflow.config import Config as MLFlowConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as PredictiveModelConfig


@dataclass
class Config:
    """Configuration for the managed run demo."""

    mlflow: MLFlowConfig
    logging: LoggingConfig
    generative_process: GenerativeProcessConfig
    persistence: PersistenceConfig
    predictive_model: PredictiveModelConfig
    experiment_name: str
    run_name: str
    seed: int
    tags: dict[str, str]
