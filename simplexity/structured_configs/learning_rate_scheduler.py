"""Learning rate scheduler configuration dataclasses."""

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
    validate_non_negative_float,
    validate_non_negative_int,
    validate_nonempty_str,
    validate_positive_float,
    validate_positive_int,
)


@dataclass
class StepLRInstanceConfig(InstanceConfig):
    """Configuration for PyTorch StepLR scheduler."""

    step_size: int = 10
    gamma: float = 0.1
    last_epoch: int = -1


def is_step_lr_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a StepLR scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.StepLR"
    return False


def validate_step_lr_instance_config(cfg: DictConfig) -> None:
    """Validate a StepLRInstanceConfig."""
    validate_instance_config(cfg)
    step_size = cfg.get("step_size")
    gamma = cfg.get("gamma")
    last_epoch = cfg.get("last_epoch")

    validate_positive_int(step_size, "StepLRInstanceConfig.step_size", is_none_allowed=True)
    validate_positive_float(gamma, "StepLRInstanceConfig.gamma", is_none_allowed=True)
    if last_epoch is not None and not isinstance(last_epoch, int):
        raise ConfigValidationError(f"StepLRInstanceConfig.last_epoch must be an int, got {type(last_epoch)}")


@dataclass
class ExponentialLRInstanceConfig(InstanceConfig):
    """Configuration for PyTorch ExponentialLR scheduler."""

    gamma: float = 0.95
    last_epoch: int = -1


def is_exponential_lr_config(cfg: DictConfig) -> bool:
    """Check if the configuration is an ExponentialLR scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.ExponentialLR"
    return False


def validate_exponential_lr_instance_config(cfg: DictConfig) -> None:
    """Validate an ExponentialLRInstanceConfig."""
    validate_instance_config(cfg)
    gamma = cfg.get("gamma")
    last_epoch = cfg.get("last_epoch")

    validate_positive_float(gamma, "ExponentialLRInstanceConfig.gamma", is_none_allowed=True)
    if last_epoch is not None and not isinstance(last_epoch, int):
        raise ConfigValidationError(f"ExponentialLRInstanceConfig.last_epoch must be an int, got {type(last_epoch)}")


@dataclass
class CosineAnnealingLRInstanceConfig(InstanceConfig):
    """Configuration for PyTorch CosineAnnealingLR scheduler."""

    T_max: int = 100
    eta_min: float = 0.0
    last_epoch: int = -1


def is_cosine_annealing_lr_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a CosineAnnealingLR scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.CosineAnnealingLR"
    return False


def validate_cosine_annealing_lr_instance_config(cfg: DictConfig) -> None:
    """Validate a CosineAnnealingLRInstanceConfig."""
    validate_instance_config(cfg)
    t_max = cfg.get("T_max")
    eta_min = cfg.get("eta_min")
    last_epoch = cfg.get("last_epoch")

    validate_positive_int(t_max, "CosineAnnealingLRInstanceConfig.T_max", is_none_allowed=True)
    validate_non_negative_float(eta_min, "CosineAnnealingLRInstanceConfig.eta_min", is_none_allowed=True)
    if last_epoch is not None and not isinstance(last_epoch, int):
        raise ConfigValidationError(
            f"CosineAnnealingLRInstanceConfig.last_epoch must be an int, got {type(last_epoch)}"
        )


@dataclass
class CosineAnnealingWarmRestartsInstanceConfig(InstanceConfig):
    """Configuration for PyTorch CosineAnnealingWarmRestarts scheduler."""

    T_0: int = 10
    T_mult: int = 1
    eta_min: float = 0.0
    last_epoch: int = -1


def is_cosine_annealing_warm_restarts_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a CosineAnnealingWarmRestarts scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    return False


def validate_cosine_annealing_warm_restarts_instance_config(cfg: DictConfig) -> None:
    """Validate a CosineAnnealingWarmRestartsInstanceConfig."""
    validate_instance_config(cfg)
    t_0 = cfg.get("T_0")
    t_mult = cfg.get("T_mult")
    eta_min = cfg.get("eta_min")
    last_epoch = cfg.get("last_epoch")

    validate_positive_int(t_0, "CosineAnnealingWarmRestartsInstanceConfig.T_0", is_none_allowed=True)
    validate_positive_int(t_mult, "CosineAnnealingWarmRestartsInstanceConfig.T_mult", is_none_allowed=True)
    validate_non_negative_float(eta_min, "CosineAnnealingWarmRestartsInstanceConfig.eta_min", is_none_allowed=True)
    if last_epoch is not None and not isinstance(last_epoch, int):
        raise ConfigValidationError(
            f"CosineAnnealingWarmRestartsInstanceConfig.last_epoch must be an int, got {type(last_epoch)}"
        )


@dataclass
class LinearLRInstanceConfig(InstanceConfig):
    """Configuration for PyTorch LinearLR scheduler (linear warmup/cooldown)."""

    start_factor: float = 1.0 / 3
    end_factor: float = 1.0
    total_iters: int = 5
    last_epoch: int = -1


def is_linear_lr_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LinearLR scheduler configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target == "torch.optim.lr_scheduler.LinearLR"
    return False


def validate_linear_lr_instance_config(cfg: DictConfig) -> None:
    """Validate a LinearLRInstanceConfig."""
    validate_instance_config(cfg)
    start_factor = cfg.get("start_factor")
    end_factor = cfg.get("end_factor")
    total_iters = cfg.get("total_iters")
    last_epoch = cfg.get("last_epoch")

    validate_positive_float(start_factor, "LinearLRInstanceConfig.start_factor", is_none_allowed=True)
    validate_positive_float(end_factor, "LinearLRInstanceConfig.end_factor", is_none_allowed=True)
    validate_positive_int(total_iters, "LinearLRInstanceConfig.total_iters", is_none_allowed=True)
    if last_epoch is not None and not isinstance(last_epoch, int):
        raise ConfigValidationError(f"LinearLRInstanceConfig.last_epoch must be an int, got {type(last_epoch)}")


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
class WindowedReduceLROnPlateauInstanceConfig(InstanceConfig):
    """Configuration for WindowedReduceLROnPlateau scheduler.

    This scheduler compares the average loss over a sliding window instead of
    individual loss values, making the patience mechanism more effective for
    noisy batch losses.
    """

    window_size: int = 10
    update_every: int = 1
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


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
    validate_instance_config(cfg)
    window_size = cfg.get("window_size")
    update_every = cfg.get("update_every")
    mode = cfg.get("mode")
    factor = cfg.get("factor")
    patience = cfg.get("patience")
    threshold = cfg.get("threshold")
    cooldown = cfg.get("cooldown")
    min_lr = cfg.get("min_lr")
    eps = cfg.get("eps")

    validate_positive_int(window_size, "WindowedReduceLROnPlateauInstanceConfig.window_size", is_none_allowed=True)
    validate_positive_int(update_every, "WindowedReduceLROnPlateauInstanceConfig.update_every", is_none_allowed=True)
    if mode is not None and mode not in ("min", "max"):
        raise ConfigValidationError(f"WindowedReduceLROnPlateauInstanceConfig.mode must be 'min' or 'max', got {mode}")
    validate_positive_float(factor, "WindowedReduceLROnPlateauInstanceConfig.factor", is_none_allowed=True)
    validate_non_negative_int(patience, "WindowedReduceLROnPlateauInstanceConfig.patience", is_none_allowed=True)
    validate_non_negative_float(threshold, "WindowedReduceLROnPlateauInstanceConfig.threshold", is_none_allowed=True)
    validate_non_negative_int(cooldown, "WindowedReduceLROnPlateauInstanceConfig.cooldown", is_none_allowed=True)
    validate_non_negative_float(min_lr, "WindowedReduceLROnPlateauInstanceConfig.min_lr", is_none_allowed=True)
    validate_non_negative_float(eps, "WindowedReduceLROnPlateauInstanceConfig.eps", is_none_allowed=True)


@dataclass
class LearningRateSchedulerConfig:
    """Base configuration for learning rate schedulers."""

    instance: InstanceConfig
    name: str | None = None


def is_lr_scheduler_target(target: str) -> bool:
    """Check if the target is a learning rate scheduler target."""
    return (
        target.startswith("torch.optim.lr_scheduler.") or target == "simplexity.lr_schedulers.WindowedReduceLROnPlateau"
    )


def is_lr_scheduler_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a learning rate scheduler config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_lr_scheduler_target(target)
    return False


def validate_lr_scheduler_config(cfg: DictConfig) -> None:
    """Validate a LearningRateSchedulerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("LearningRateSchedulerConfig.instance must be a DictConfig")
    name = cfg.get("name")

    if is_step_lr_config(instance):
        validate_step_lr_instance_config(instance)
    elif is_exponential_lr_config(instance):
        validate_exponential_lr_instance_config(instance)
    elif is_cosine_annealing_lr_config(instance):
        validate_cosine_annealing_lr_instance_config(instance)
    elif is_cosine_annealing_warm_restarts_config(instance):
        validate_cosine_annealing_warm_restarts_instance_config(instance)
    elif is_linear_lr_config(instance):
        validate_linear_lr_instance_config(instance)
    elif is_reduce_lr_on_plateau_config(instance):
        validate_reduce_lr_on_plateau_instance_config(instance)
    elif is_windowed_reduce_lr_on_plateau_config(instance):
        validate_windowed_reduce_lr_on_plateau_instance_config(instance)
    else:
        validate_instance_config(instance)
        if not is_lr_scheduler_config(instance):
            raise ConfigValidationError("LearningRateSchedulerConfig.instance must be a learning rate scheduler target")
    validate_nonempty_str(name, "LearningRateSchedulerConfig.name", is_none_allowed=True)
