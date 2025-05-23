from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.configs.evaluation.config import (
    Config as ValidationConfig,
)
from simplexity.configs.evaluation.config import (
    validate_config as validate_validation_config,
)
from simplexity.configs.generative_process.config import Config as DataGeneratorConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.predictive_model.config import validate_config as validate_model_config
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.training.config import validate_config as validate_training_config


@dataclass
class Config(DictConfig):
    """Configuration for the experiment."""

    training_data_generator: DataGeneratorConfig
    validation_data_generator: DataGeneratorConfig | None
    predictive_model: ModelConfig
    persistence: PersistenceConfig | None
    logging: LoggingConfig | None
    training: TrainingConfig
    validation: ValidationConfig | None

    seed: int
    experiment_name: str
    run_name: str


def validation_required(cfg: Config) -> bool:
    """Check if validation is required."""
    return (
        cfg.training.validate_every is not None
        and cfg.training.validate_every > 0
        and cfg.training.validate_every <= cfg.training.num_steps
    )


def persistence_required(cfg: Config) -> bool:
    """Check if persistence is required."""
    return cfg.predictive_model.load_checkpoint_step is not None or (
        cfg.training.checkpoint_every is not None
        and cfg.training.checkpoint_every > 0
        and cfg.training.checkpoint_every <= cfg.training.num_steps
    )


def logging_required(cfg: Config) -> bool:
    """Check if logging is required."""
    if (
        cfg.training.log_every is not None
        and cfg.training.log_every > 0
        and cfg.training.log_every <= cfg.training.num_steps
    ):
        return True
    return bool(
        validation_required(cfg)
        and cfg.validation
        and cfg.validation.log_every is not None
        and cfg.validation.log_every > 0
        and cfg.validation.log_every <= cfg.validation.num_steps
    )


def validate_config(cfg: Config) -> None:
    """Validate the configuration."""
    validate_model_config(cfg.predictive_model)
    validate_training_config(cfg.training)
    if validation_required(cfg):
        assert cfg.validation is not None, "Validation is required but not configured"
        validate_validation_config(cfg.validation)
        assert cfg.validation_data_generator is not None, "Validation data generator is required but not configured"
    else:
        assert cfg.validation is None, "Validation is configured but not required"
        assert cfg.validation_data_generator is None, "Validation data generator is configured but not required"

    if persistence_required(cfg):
        assert cfg.persistence is not None, "Persistence is required but not configured"
    else:
        assert not cfg.persistence, "Persistence is configured but not required"

    if cfg.logging:
        assert logging_required(cfg), "Logging is configured but not required"
    else:
        assert not logging_required(cfg), "Logging is required but not configured"
