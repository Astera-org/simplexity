"""Base configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.mlflow import MLFlowConfig, validate_mlflow_config
from simplexity.structured_configs.validation import _validate_mapping, _validate_non_negative_int


@dataclass
class BaseConfig:
    """Base configuration for all components."""

    seed: int | None = None
    tags: dict[str, str] | None = None
    mlflow: MLFlowConfig | None = None


def validate_base_config(cfg: DictConfig) -> None:
    """Validate a BaseConfig.

    Args:
        cfg: A DictConfig with seed, tags, and mlflow fields (from Hydra).
    """
    seed = cfg.get("seed")
    tags = cfg.get("tags")
    mlflow = cfg.get("mlflow")

    _validate_non_negative_int(seed, "BaseConfig.seed", is_none_allowed=True)
    _validate_mapping(tags, "BaseConfig.tags", key_type=str, value_type=str, is_none_allowed=True)
    if mlflow is not None:
        if not isinstance(mlflow, DictConfig):
            raise ConfigValidationError("BaseConfig.mlflow must be a MLFlowConfig")
        validate_mlflow_config(mlflow)

