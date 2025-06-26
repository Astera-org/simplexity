from dataclasses import dataclass
from typing import Literal


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
