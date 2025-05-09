from dataclasses import dataclass

from simplexity.configs.evaluation.config import Config as ValidationConfig
from simplexity.configs.generative_process.config import Config as DataGeneratorConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.state_sampler.config import Config as StateSamplerConfig
from simplexity.configs.training.config import Config as TrainingConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    training_data_generator: DataGeneratorConfig
    validation_data_generator: DataGeneratorConfig | None
    training_state_sampler: StateSamplerConfig | None
    validation_state_sampler: StateSamplerConfig | None
    predictive_model: ModelConfig
    persistence: PersistenceConfig | None
    logging: LoggingConfig | None
    training: TrainingConfig
    validation: ValidationConfig | None

    seed: int
    experiment_name: str
    run_name: str
