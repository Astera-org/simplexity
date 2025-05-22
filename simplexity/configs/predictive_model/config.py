from dataclasses import dataclass
from typing import Literal

TARGETS = Literal["simplexity.predictive_models.gru_rnn.build_gru_rnn"]


@dataclass
class ModelInstanceConfig:
    """Configuration for the model instance."""

    _target_: TARGETS
    vocab_size: int


@dataclass
class GRURNNConfig(ModelInstanceConfig):
    """Configuration for GRU RNN model."""

    embedding_size: int
    num_layers: int
    hidden_size: int
    seed: int


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: str
    instance: ModelInstanceConfig
    load_checkpoint_step: int | None
