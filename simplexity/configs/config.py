from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as ProcessConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.train.config import Config as TrainConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    generative_process: ProcessConfig
    predictive_model: ModelConfig
    persistence: PersistenceConfig
    logging: LoggingConfig
    train: TrainConfig

    seed: int
