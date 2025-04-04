from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as DataGeneratorConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.validation.config import Config as ValidationConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    training_data_generator: DataGeneratorConfig
    validation_data_generator: DataGeneratorConfig
    predictive_model: ModelConfig
    persistence: PersistenceConfig
    logging: LoggingConfig
    training: TrainingConfig
    validation: ValidationConfig

    seed: int
    experiment_name: str
    run_name: str
