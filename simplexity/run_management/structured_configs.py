"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

from dataclasses import dataclass
from typing import Literal

from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.configs.instance_config import InstanceConfig
from simplexity.predictive_models.predictive_model import is_predictive_model_target

# ============================================================================
# Base Config
# ============================================================================

# InstanceConfig is already defined in configs/instance_config.py and imported above


# ============================================================================
# Optimizer Config
# ============================================================================


@dataclass
class OptimizerConfig:
    """Base configuration for optimizers."""

    name: str
    instance: InstanceConfig


def is_pytorch_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.")
    return False


# ============================================================================
# Logging Config
# ============================================================================


@dataclass
class LoggingConfig:
    """Base configuration for logging."""

    name: str
    instance: InstanceConfig


def is_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LoggingInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.logging.")
    return False


# ============================================================================
# Persistence Config
# ============================================================================


@dataclass
class PersistenceConfig:
    """Base configuration for persistence."""

    name: str
    instance: InstanceConfig


def is_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PersistenceInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.persistence.")
    return False


# ============================================================================
# Generative Process Config
# ============================================================================


@dataclass
class GenerativeProcessConfig:
    """Base configuration for generative processes."""

    name: str
    instance: InstanceConfig
    base_vocab_size: int = MISSING
    bos_token: int | None = MISSING
    eos_token: int | None = MISSING
    vocab_size: int = MISSING


def is_generative_process_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generative process config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.generative_processes.")
    return False


# ============================================================================
# Predictive Model Configs
# ============================================================================


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
class ModelConfig:
    """Base configuration for predictive models."""

    name: str
    instance: InstanceConfig
    load_checkpoint_step: int | None = None


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate the configuration."""
    if cfg.load_checkpoint_step is not None:
        assert cfg.load_checkpoint_step >= 0, "Load checkpoint step must be non-negative"


def is_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_predictive_model_target(target)
    return False


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"


# ============================================================================
# Training Config
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for the training process."""

    seed: int
    batch_size: int
    num_steps: int
    optimizer: OptimizerConfig
    sequence_len: int = MISSING
    log_every: int | None = None
    validate_every: int | None = None
    checkpoint_every: int | None = None


def validate_training_config(cfg: TrainingConfig) -> None:
    """Validate the configuration."""
    assert cfg.sequence_len > 0, "Sequence length must be greater than 0"
    assert cfg.batch_size > 0, "Batch size must be greater than 0"
    assert cfg.num_steps > 0, "Number of steps must be greater than 0"
    if cfg.log_every is not None:
        assert cfg.log_every > 0, "Log every must be greater than 0"
        assert cfg.log_every <= cfg.num_steps, "Log every must be less than or equal to number of steps"
    if cfg.validate_every is not None:
        assert cfg.validate_every > 0, "Validate every must be greater than 0"
        assert cfg.validate_every <= cfg.num_steps, "Validate every must be less than or equal to number of steps"
    if cfg.checkpoint_every is not None:
        assert cfg.checkpoint_every > 0, "Checkpoint every must be greater than 0"
        assert cfg.checkpoint_every <= cfg.num_steps, "Checkpoint every must be less than or equal to number of steps"


# ============================================================================
# Validation Config
# ============================================================================


@dataclass
class ValidationConfig:
    """Configuration for the validation."""

    seed: int
    sequence_len: int
    batch_size: int
    num_steps: int
    log_every: int | None


def validate_validation_config(cfg: ValidationConfig) -> None:
    """Validate the configuration."""
    assert cfg.sequence_len > 0, "Sequence length must be greater than 0"
    assert cfg.batch_size > 0, "Batch size must be greater than 0"
    assert cfg.num_steps > 0, "Number of steps must be greater than 0"
    if cfg.log_every is not None:
        assert cfg.log_every > 0, "Log every must be greater than 0"
        assert cfg.log_every <= cfg.num_steps, "Log every must be less than or equal to number of steps"


# ============================================================================
# MLflow Config
# ============================================================================


@dataclass
class MLFlowConfig:
    """Configuration for MLflow."""

    experiment_name: str
    run_name: str
    tracking_uri: str
    registry_uri: str
    downgrade_unity_catalog: bool


# ============================================================================
# Main Config
# ============================================================================


@dataclass
class MainConfig:
    """Configuration for the experiment."""

    training_data_generator: GenerativeProcessConfig
    validation_data_generator: GenerativeProcessConfig | None
    predictive_model: ModelConfig
    persistence: PersistenceConfig | None
    logging: LoggingConfig | None
    training: TrainingConfig
    validation: ValidationConfig | None

    seed: int
    experiment_name: str
    run_name: str


def validation_required(cfg: MainConfig) -> bool:
    """Check if validation is required."""
    return (
        cfg.training.validate_every is not None
        and cfg.training.validate_every > 0
        and cfg.training.validate_every <= cfg.training.num_steps
    )


def persistence_required(cfg: MainConfig) -> bool:
    """Check if persistence is required."""
    return cfg.predictive_model.load_checkpoint_step is not None or (
        cfg.training.checkpoint_every is not None
        and cfg.training.checkpoint_every > 0
        and cfg.training.checkpoint_every <= cfg.training.num_steps
    )


def logging_required(cfg: MainConfig) -> bool:
    """Check if logging is required."""
    if (
        cfg.training.log_every is not None
        and cfg.training.log_every > 0
        and cfg.training.log_every <= cfg.training.num_steps
    ):
        return True
    return bool(
        validation_required(cfg)
        and cfg.validation
        and cfg.validation.log_every is not None
        and cfg.validation.log_every > 0
        and cfg.validation.log_every <= cfg.validation.num_steps
    )


def validate_config(cfg: MainConfig) -> None:
    """Validate the configuration."""
    validate_model_config(cfg.predictive_model)
    validate_training_config(cfg.training)
    if validation_required(cfg):
        assert cfg.validation is not None, "Validation is required but not configured"
        validate_validation_config(cfg.validation)
        assert cfg.validation_data_generator is not None, "Validation data generator is required but not configured"
    else:
        assert cfg.validation is None, "Validation is configured but not required"
        assert cfg.validation_data_generator is None, "Validation data generator is configured but not required"

    if persistence_required(cfg):
        assert cfg.persistence is not None, "Persistence is required but not configured"
    else:
        assert not cfg.persistence, "Persistence is configured but not required"

    if cfg.logging:
        assert logging_required(cfg), "Logging is configured but not required"
    else:
        assert not logging_required(cfg), "Logging is required but not configured"
