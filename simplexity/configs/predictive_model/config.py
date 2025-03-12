from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelInstanceConfig:
    """Configuration for the model instance."""

    _target_: Literal["simplexity.predictive_models.rnn.build_rnn"]


@dataclass
class RNNConfig(ModelInstanceConfig):
    """Configuration for RNN model."""

    _target_: Literal["simplexity.predictive_models.rnn.build_rnn"]
    in_size: int
    hidden_size: int
    num_layers: int
    out_size: int
    seed: int


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: Literal["rnn"]
    instance: ModelInstanceConfig
    weights_filename: str
    load_weights: bool
