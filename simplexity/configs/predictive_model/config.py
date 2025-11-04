from dataclasses import dataclass
from typing import Literal

from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.configs.instance_config import InstanceConfig
from simplexity.predictive_models.predictive_model import is_predictive_model_target


@dataclass
class GRURNNConfig(InstanceConfig):
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
class HookedTransformerConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: str
    instance: InstanceConfig
    load_checkpoint_step: int | None = None


def validate_config(cfg: Config) -> None:
    """Validate the configuration."""
    if cfg.load_checkpoint_step is not None:
        assert cfg.load_checkpoint_step >= 0, "Load checkpoint step must be non-negative"


def is_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PersistenceInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_predictive_model_target(target)
    return False


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"
