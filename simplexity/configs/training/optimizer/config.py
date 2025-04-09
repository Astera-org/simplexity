from dataclasses import dataclass
from typing import Literal, Optional, Any


@dataclass
class OptimizerInstanceConfig:
    """Configuration for the optimizer instance."""

    _target_: Literal[
        "optax.adam",
        "optax.adamw",
    ]

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
class AdamWConfig(OptimizerInstanceConfig):
    """Configuration for AdamW optimizer."""

    learning_rate: float
    b1: float
    b2: float
    eps: float
    eps_root: float
    # dtype to be used with accumulators.  unsupported here
    # mu_dtype: Optional[Any] 
    weight_decay: float
    # mask for selective updating of parameters, unsupported here
    # mask: Any 
    nesterov: bool

@dataclass
class Config:
    """Base configuration for predictive models."""

    name: Literal["adam", "adamw"]
    instance: OptimizerInstanceConfig

