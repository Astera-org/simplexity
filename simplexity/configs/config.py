from dataclasses import dataclass

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
class Config:
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
    from omegaconf import DictConfig

    if (
        cfg.training.log_every is not None
        and cfg.training.log_every > 0
        and cfg.training.log_every <= cfg.training.num_steps
    ):
        return True

    has_validation = isinstance(cfg, DictConfig) and "validation" in cfg
    validation_value = cfg.validation if has_validation else None

    return bool(
        validation_required(cfg)
        and validation_value
        and validation_value.log_every is not None
        and validation_value.log_every > 0
        and validation_value.log_every <= validation_value.num_steps
    )


def validate_config(cfg: Config) -> None:
    """Validate the configuration."""
    from omegaconf import DictConfig

    validate_model_config(cfg.predictive_model)
    validate_training_config(cfg.training)

    # Handle validation config (may not be present in config file)
    has_validation = isinstance(cfg, DictConfig) and "validation" in cfg
    validation_value = cfg.validation if has_validation else None

    if validation_required(cfg):
        assert validation_value is not None, "Validation is required but not configured"
        validate_validation_config(validation_value)
        has_validation_generator = isinstance(cfg, DictConfig) and "validation_data_generator" in cfg
        validation_generator_value = cfg.validation_data_generator if has_validation_generator else None
        assert validation_generator_value is not None, "Validation data generator is required but not configured"
    else:
        assert validation_value is None, "Validation is configured but not required"
        has_validation_generator = isinstance(cfg, DictConfig) and "validation_data_generator" in cfg
        validation_generator_value = cfg.validation_data_generator if has_validation_generator else None
        assert validation_generator_value is None, "Validation data generator is configured but not required"

    # Handle persistence config (may not be present in config file)
    has_persistence = isinstance(cfg, DictConfig) and "persistence" in cfg
    persistence_value = cfg.persistence if has_persistence else None

    if persistence_required(cfg):
        assert persistence_value is not None, "Persistence is required but not configured"
    else:
        assert not persistence_value, "Persistence is configured but not required"

    # Handle logging config (may not be present in config file)
    has_logging = isinstance(cfg, DictConfig) and "logging" in cfg
    logging_value = cfg.logging if has_logging else None

    if logging_value:
        assert logging_required(cfg), "Logging is configured but not required"
    else:
        assert not logging_required(cfg), "Logging is required but not configured"
