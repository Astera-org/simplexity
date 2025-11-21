"""Optimizer configuration dataclasses."""

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
from simplexity.structured_configs.instance import InstanceConfig, _validate_instance_config
from simplexity.structured_configs.validation import (
    _validate_bool,
    _validate_non_negative_float,
    _validate_nonempty_str,
    _validate_positive_float,
)


@dataclass
class AdamInstanceConfig(InstanceConfig):
    """Configuration for the Adam optimizer."""

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


def is_pytorch_adam_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.adam")
    return False


def validate_adam_instance_config(cfg: DictConfig) -> None:
    """Validate an AdamInstanceConfig.

    Args:
        cfg: A DictConfig with AdamInstanceConfig fields (from Hydra).
    """
    _validate_instance_config(cfg)
    lr = cfg.get("lr")
    betas = cfg.get("betas")
    eps = cfg.get("eps")
    weight_decay = cfg.get("weight_decay")
    amsgrad = cfg.get("amsgrad")

    _validate_positive_float(lr, "AdamInstanceConfig.lr", is_none_allowed=True)
    if betas is not None:
        if len(betas) != 2:
            raise ConfigValidationError(f"AdamInstanceConfig.betas must have length 2, got {len(betas)}")
        _validate_non_negative_float(betas[0], "AdamInstanceConfig.betas[0]")
        _validate_non_negative_float(betas[1], "AdamInstanceConfig.betas[1]")
    _validate_non_negative_float(eps, "AdamInstanceConfig.eps", is_none_allowed=True)
    _validate_non_negative_float(weight_decay, "AdamInstanceConfig.weight_decay", is_none_allowed=True)
    _validate_bool(amsgrad, "AdamInstanceConfig.amsgrad", is_none_allowed=True)


@dataclass
class OptimizerConfig:
    """Base configuration for optimizers."""

    instance: InstanceConfig
    name: str | None = None


def is_optimizer_target(target: str) -> bool:
    """Check if the target is an optimizer target."""
    return target.startswith("torch.optim.") or target.startswith("optax.")


def is_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a optimizer config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_optimizer_target(target)
    return False


def is_pytorch_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.")
    return False


def validate_optimizer_config(cfg: DictConfig) -> None:
    """Validate an OptimizerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("OptimizerConfig.instance must be a DictConfig")
    target = instance.get("_target_")
    name = cfg.get("name")

    _validate_instance_config(instance)
    if not is_optimizer_target(target):
        raise ConfigValidationError(f"OptimizerConfig.instance._target_ must be an optimizer target, got {target}")
    elif is_pytorch_adam_optimizer_config(instance):
        validate_adam_instance_config(instance)
    _validate_nonempty_str(name, "OptimizerConfig.name", is_none_allowed=True)

