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
class Config:
    """Base configuration for predictive models."""

    name: Literal["adam"]
    instance: OptimizerInstanceConfig
