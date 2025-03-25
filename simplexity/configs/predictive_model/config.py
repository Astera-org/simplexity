from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelInstanceConfig:
    """Configuration for the model instance."""

    _target_: Literal["simplexity.predictive_models.rnn.build_rnn"]
    vocab_size: int


@dataclass
class GRURNNConfig(ModelInstanceConfig):
    """Configuration for GRU RNN model."""

    num_layers: int
    hidden_size: int
    seed: int


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: Literal["gru_rnn"]
    instance: ModelInstanceConfig
    load_checkpoint_name: str
