"""Activatiob tracker configuration dataclasses."""

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
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import (
    validate_nonempty_str,
)


@dataclass
class ActivationTrackerConfig:
    """Configuration for activation tracker."""

    instance: InstanceConfig
    name: str | None = None


def is_activation_analysis_target(target: str) -> bool:
    """Check if the target is an activation analysis target."""
    return target.startswith("simplexity.activations.")


def is_activation_tracker_target(target: str) -> bool:
    """Check if the target is an activation tracker target."""
    return target == "simplexity.activations.activation_tracker.ActivationTracker"


def validate_activation_analysis_config(cfg: DictConfig) -> None:
    """Validate an ActivationAnalysisConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("ActivationAnalysisConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_activation_analysis_target(target):
        raise ConfigValidationError(
            f"ActivationAnalysisConfig.instance._target_ must be an activation analysis target, got {target}"
        )
    validate_nonempty_str(cfg.get("name"), "ActivationAnalysisConfig.name", is_none_allowed=True)


def validate_activation_tracker_config(cfg: DictConfig) -> None:
    """Validate an ActivationTrackerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("ActivationTrackerConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_activation_tracker_target(target):
        raise ConfigValidationError(
            f"ActivationTrackerConfig.instance._target_ must be ActivationTracker, got {target}"
        )

    analyses = instance.get("analyses")
    if analyses is None:
        raise ConfigValidationError("ActivationTrackerConfig.instance.analyses is required")

    if not isinstance(analyses, DictConfig):
        raise ConfigValidationError("ActivationTrackerConfig.instance.analyses must be a dictionary")

    for key, analysis_config in analyses.items():
        if not isinstance(analysis_config, DictConfig):
            raise ConfigValidationError(f"ActivationTrackerConfig.instance.analyses[{key}] must be a config dict")
        instance_cfg = analysis_config.get("instance")
        if not isinstance(instance_cfg, DictConfig):
            raise ConfigValidationError(
                f"ActivationTrackerConfig.instance.analyses[{key}] must specify an InstanceConfig"
            )
        target = instance_cfg.get("_target_", None)
        if not is_activation_analysis_target(target):
            raise ConfigValidationError(
                f"ActivationTrackerConfig.instance.analyses[{key}] must target an activation analysis, got {target}"
            )

    validate_nonempty_str(cfg.get("name"), "ActivationTrackerConfig.name", is_none_allowed=True)
