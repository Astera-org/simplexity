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
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.structured_configs.mlflow import MLFlowConfig, validate_mlflow_config
from simplexity.structured_configs.validation import validate_mapping, validate_non_negative_int
from simplexity.utils.config_utils import dynamic_resolve


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

    validate_non_negative_int(seed, "BaseConfig.seed", is_none_allowed=True)
    validate_mapping(tags, "BaseConfig.tags", key_type=str, value_type=str, is_none_allowed=True)
    if mlflow is not None:
        if not isinstance(mlflow, DictConfig):
            raise ConfigValidationError("BaseConfig.mlflow must be a MLFlowConfig")
        validate_mlflow_config(mlflow)


@dynamic_resolve
def resolve_base_config(cfg: DictConfig, *, strict: bool, seed: int = 42) -> None:
    """Resolve the BaseConfig by setting default values and logging mismatches.

    This function sets default seed and strict tag values if not present in the config.
    If values are already set but don't match the provided parameters, it logs
    a warning and overrides them.

    Args:
        cfg: A DictConfig with seed and tags fields (from Hydra).
        strict: Whether strict mode is enabled. Used to set tags.strict.
        seed: The random seed to use. Defaults to 42.
    """
    if cfg.get("seed") is None:
        cfg.seed = seed
    else:
        seed_tag: int = cfg.get("seed")
        if seed_tag != seed:
            SIMPLEXITY_LOGGER.warning("Seed tag set to '%s', but seed is '%s'. Overriding seed tag.", seed_tag, seed)
            cfg.seed = seed

    if cfg.get("tags") is None:
        cfg.tags = DictConfig({"strict": str(strict).lower()})
    else:
        tags: DictConfig = cfg.get("tags")
        strict_value = str(strict).lower()
        if tags.get("strict") is None:
            tags.strict = strict_value
        else:
            strict_tag: str = tags.get("strict")
            if strict_tag.lower() != strict_value:
                SIMPLEXITY_LOGGER.warning(
                    "Strict tag set to '%s', but strict mode is '%s'. Overriding strict tag.", strict_tag, strict_value
                )
                tags.strict = strict_value
