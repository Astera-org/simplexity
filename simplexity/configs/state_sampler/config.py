from dataclasses import dataclass
from typing import Any

import jax


@dataclass
class StateSamplerInstanceConfig:
    """Configuration for a state sampler."""

    _target_: str


@dataclass
class FixedStateSamplerInstanceConfig(StateSamplerInstanceConfig):
    """Configuration for a fixed state sampler."""

    state: jax.Array


@dataclass
class NonergodicStateSamplerInstanceConfig(StateSamplerInstanceConfig):
    """Configuration for a nonergodic state sampler."""

    process_names: list[str]
    process_kwargs: list[dict[str, Any]]
    mixture_weights: list[float]


@dataclass
class Config:
    """Base configuration for a state sampler."""

    name: str
    instance: StateSamplerInstanceConfig
