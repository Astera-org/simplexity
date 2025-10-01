"""Minimal configuration for basic training examples.

This config is designed for simple training scenarios where you only need:
- A data generator
- A model
- Training settings
- Optional logging

This is a pedagogical example showing the minimal config structure needed for
basic experiments. For full-featured experiments with validation, checkpointing,
and complex workflows, use simplexity.configs.config.Config instead.
"""

from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as DataGeneratorConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.predictive_model.config import validate_config as validate_model_config
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.training.config import validate_config as validate_training_config


@dataclass
class Config:
    """Minimal configuration for basic training examples."""

    training_data_generator: DataGeneratorConfig
    predictive_model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig | None

    seed: int
    experiment_name: str
    run_name: str


def validate_config(cfg: Config) -> None:
    """Validate the minimal configuration."""
    validate_model_config(cfg.predictive_model)
    validate_training_config(cfg.training)
