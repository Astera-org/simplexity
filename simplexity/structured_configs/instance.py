"""Instance configuration dataclasses."""

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
from simplexity.structured_configs.validation import _validate_nonempty_str


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str


def validate_instance_config(cfg: DictConfig, expected_target: str | None = None) -> None:
    """Validate an InstanceConfig.

    Args:
        cfg: A DictConfig with an _target_ field (from Hydra).
        expected_target: The expected target, if any.
    """
    target = cfg.get("_target_", None)

    _validate_nonempty_str(target, "InstanceConfig._target_")
    if expected_target is not None and target != expected_target:
        raise ConfigValidationError(f"InstanceConfig._target_ must be {expected_target}, got {target}")
