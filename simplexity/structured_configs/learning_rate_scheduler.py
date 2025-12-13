"""Learning rate scheduler configuration dataclasses."""

from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import (
    validate_non_negative_float,
    validate_non_negative_int,
    validate_nonempty_str,
    validate_positive_float,
    validate_positive_int,
)


@dataclass
class ReduceLROnPlateauInstanceConfig(InstanceConfig):
    """Configuration for PyTorch ReduceLROnPlateau scheduler."""

    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


@dataclass
class WindowedReduceLROnPlateauInstanceConfig(ReduceLROnPlateauInstanceConfig):
    """Configuration for WindowedReduceLROnPlateau scheduler.

    This scheduler compares the average loss over a sliding window instead of
    individual loss values, making the patience mechanism more effective for
    noisy batch losses.

    Inherits all fields from ReduceLROnPlateauInstanceConfig and adds:
    - window_size: Size of the sliding window for loss averaging
    - update_every: Frequency of scheduler updates (steps between updates)
    """

    window_size: int = 10
    update_every: int = 1


def is_reduce_lr_on_plateau_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a ReduceLROnPlateau scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.ReduceLROnPlateau"
    return False


def validate_reduce_lr_on_plateau_instance_config(cfg: DictConfig) -> None:
    """Validate a ReduceLROnPlateauInstanceConfig."""
    validate_instance_config(cfg)
    mode = cfg.get("mode")
    factor = cfg.get("factor")
    patience = cfg.get("patience")
    threshold = cfg.get("threshold")
    cooldown = cfg.get("cooldown")
    min_lr = cfg.get("min_lr")
    eps = cfg.get("eps")

    if mode is not None and mode not in ("min", "max"):
        raise ConfigValidationError(f"ReduceLROnPlateauInstanceConfig.mode must be 'min' or 'max', got {mode}")
    validate_positive_float(factor, "ReduceLROnPlateauInstanceConfig.factor", is_none_allowed=True)
    validate_non_negative_int(patience, "ReduceLROnPlateauInstanceConfig.patience", is_none_allowed=True)
    validate_non_negative_float(threshold, "ReduceLROnPlateauInstanceConfig.threshold", is_none_allowed=True)
    validate_non_negative_int(cooldown, "ReduceLROnPlateauInstanceConfig.cooldown", is_none_allowed=True)
    validate_non_negative_float(min_lr, "ReduceLROnPlateauInstanceConfig.min_lr", is_none_allowed=True)
    validate_non_negative_float(eps, "ReduceLROnPlateauInstanceConfig.eps", is_none_allowed=True)


def is_windowed_reduce_lr_on_plateau_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a WindowedReduceLROnPlateau scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "simplexity.lr_schedulers.WindowedReduceLROnPlateau"
    return False


def validate_windowed_reduce_lr_on_plateau_instance_config(cfg: DictConfig) -> None:
    """Validate a WindowedReduceLROnPlateauInstanceConfig."""
    validate_reduce_lr_on_plateau_instance_config(cfg)
    window_size = cfg.get("window_size")
    update_every = cfg.get("update_every")

    validate_positive_int(window_size, "WindowedReduceLROnPlateauInstanceConfig.window_size", is_none_allowed=True)
    validate_positive_int(update_every, "WindowedReduceLROnPlateauInstanceConfig.update_every", is_none_allowed=True)


@dataclass
class LearningRateSchedulerConfig:
    """Base configuration for learning rate schedulers."""

    instance: InstanceConfig
    name: str | None = None


def is_lr_scheduler_target(target: str) -> bool:
    """Check if the target is a supported learning rate scheduler target."""
    return target in (
        "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "simplexity.lr_schedulers.WindowedReduceLROnPlateau",
    )


def is_lr_scheduler_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a plateau-based learning rate scheduler config."""
    return is_reduce_lr_on_plateau_config(cfg) or is_windowed_reduce_lr_on_plateau_config(cfg)


def validate_lr_scheduler_config(cfg: DictConfig) -> None:
    """Validate a LearningRateSchedulerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("LearningRateSchedulerConfig.instance must be a DictConfig")
    name = cfg.get("name")

    if is_reduce_lr_on_plateau_config(instance):
        validate_reduce_lr_on_plateau_instance_config(instance)
    elif is_windowed_reduce_lr_on_plateau_config(instance):
        validate_windowed_reduce_lr_on_plateau_instance_config(instance)
    else:
        validate_instance_config(instance)
        if not is_lr_scheduler_config(instance):
            raise ConfigValidationError(
                "LearningRateSchedulerConfig.instance must be ReduceLROnPlateau or WindowedReduceLROnPlateau"
            )
    validate_nonempty_str(name, "LearningRateSchedulerConfig.name", is_none_allowed=True)
