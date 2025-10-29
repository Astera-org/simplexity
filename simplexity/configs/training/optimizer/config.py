from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig


@dataclass
class OptimizerInstanceConfig:
    """Configuration for the optimizer instance."""

    _target_: Literal["optax.adam"]


@dataclass
class AdamConfig(OptimizerInstanceConfig):
    """Configuration for Adam optimizer."""

    learning_rate: float
    b1: float
    b2: float
    eps: float
    eps_root: float
    nesterov: bool


@dataclass
class PytorchOptimizerInstanceConfig:
    """Configuration for PyTorch optimizer instance."""

    _target_: Literal["torch.optim.AdamW", "torch.optim.Adam", "torch.optim.SGD"]


@dataclass
class PytorchAdamConfig(PytorchOptimizerInstanceConfig):
    """Configuration for PyTorch Adam optimizer."""

    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: Literal["adam", "pytorch_adam"]
    instance: OptimizerInstanceConfig | PytorchOptimizerInstanceConfig


def is_pytorch_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target: str | None = cfg.get("_target_", None)
    if target is None:
        return False
    return target.startswith("torch.optim.")
