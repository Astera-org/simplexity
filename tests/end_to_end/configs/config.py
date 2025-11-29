"""Configuration for the managed run demo."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

from simplexity.structured_configs.generative_process import GenerativeProcessConfig
from simplexity.structured_configs.logging import LoggingConfig
from simplexity.structured_configs.mlflow import MLFlowConfig
from simplexity.structured_configs.optimizer import OptimizerConfig
from simplexity.structured_configs.persistence import PersistenceConfig
from simplexity.structured_configs.predictive_model import PredictiveModelConfig


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_steps: int
    batch_size: int
    sequence_len: int
    log_every: int
    checkpoint_every: int
    evaluate_every: int


@dataclass
class Config:
    """Configuration for the managed run demo."""

    device: str
    mlflow: MLFlowConfig
    logging: LoggingConfig
    generative_process: GenerativeProcessConfig
    persistence: PersistenceConfig
    predictive_model: PredictiveModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    experiment_name: str
    run_name: str
    seed: int
    tags: dict[str, str]
