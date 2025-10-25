from dataclasses import dataclass
from typing import Literal

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ModelInstanceConfig:
    """Configuration for the model instance."""

    _target_: Literal["simplexity.predictive_models.gru_rnn.build_gru_rnn", "transformer_lens.HookedTransformer"]


@dataclass
class GRURNNConfig(ModelInstanceConfig):
    """Configuration for GRU RNN model."""

    embedding_size: int
    num_layers: int
    hidden_size: int
    seed: int
    vocab_size: int = MISSING


@dataclass
class HookedTransformerConfigConfig:
    """Configuration for HookedTransformerConfig."""

    _target_: Literal["transformer_lens.HookedTransformerConfig"]
    d_model: int
    d_head: int
    n_heads: int
    n_layers: int
    n_ctx: int
    d_mlp: int
    act_fn: str | None
    normalization_type: str | None
    device: str | None
    seed: int
    d_vocab: int = MISSING


@dataclass
class HookedTransformerConfig(ModelInstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: str
    instance: ModelInstanceConfig
    load_checkpoint_step: int | None = None


def validate_config(cfg: Config) -> None:
    """Validate the configuration."""
    if cfg.load_checkpoint_step is not None:
        assert cfg.load_checkpoint_step >= 0, "Load checkpoint step must be non-negative"


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"
