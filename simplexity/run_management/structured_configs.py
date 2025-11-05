"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from omegaconf import MISSING, DictConfig, OmegaConf

# ============================================================================
# Validation Helpers
# ============================================================================


def validate_optional_name(name: str | None, config_name: str, field_name: str = "name") -> None:
    """Validate that a name field is either None or a non-empty string.

    Args:
        name: The name value to validate
        config_name: The name of the config class (for error messages)
        field_name: The name of the field (defaults to "name")
    """
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"{config_name}.{field_name} must be None or a non-empty string")


# ============================================================================
# Base Config
# ============================================================================


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str

    def __init__(self, _target_: str, **kwargs: Any):
        self._target_ = _target_
        for key, value in kwargs.items():
            setattr(self, key, value)


def validate_instance_config(cfg: InstanceConfig) -> None:
    """Validate an InstanceConfig."""
    if not isinstance(cfg._target_, str):
        raise ValueError(f"InstanceConfig._target_ must be a string, got {type(cfg._target_)}")
    if not cfg._target_.strip():
        raise ValueError("InstanceConfig._target_ cannot be empty or whitespace")


# ============================================================================
# Optimizer Config
# ============================================================================


@dataclass
class OptimizerConfig:
    """Base configuration for optimizers."""

    instance: InstanceConfig
    name: str | None = None


def is_optimizer_target(target: str) -> bool:
    """Check if the target is an optimizer target."""
    return target.startswith("torch.optim.") or target.startswith("optax.")


def is_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a optimizer config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_optimizer_target(target)
    return False


def is_pytorch_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.")
    return False


def validate_optimizer_config(cfg: OptimizerConfig) -> None:
    """Validate an OptimizerConfig."""
    validate_instance_config(cfg.instance)
    if not is_optimizer_target(cfg.instance._target_):
        raise ValueError(f"OptimizerConfig.instance._target_ must be an optimizer target, got {cfg.instance._target_}")
    validate_optional_name(cfg.name, "OptimizerConfig")


# ============================================================================
# Logging Config
# ============================================================================


@dataclass
class LoggingConfig:
    """Base configuration for logging."""

    instance: InstanceConfig
    name: str | None = None


def is_logger_target(target: str) -> bool:
    """Check if the target is a logger target."""
    return target.startswith("simplexity.logging.")


def is_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LoggingInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_logger_target(target)
    return False


def validate_logging_config(cfg: LoggingConfig) -> None:
    """Validate a LoggingConfig."""
    validate_instance_config(cfg.instance)
    if not is_logger_target(cfg.instance._target_):
        raise ValueError(f"LoggingConfig.instance._target_ must be a logger target, got {cfg.instance._target_}")
    validate_optional_name(cfg.name, "LoggingConfig")


# ============================================================================
# Persistence Config
# ============================================================================


@dataclass
class PersistenceConfig:
    """Base configuration for persistence."""

    instance: InstanceConfig
    name: str | None = None


def is_model_persister_target(target: str) -> bool:
    """Check if the target is a model persister target."""
    return target.startswith("simplexity.persistence.")


def is_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PersistenceInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_model_persister_target(target)
    return False


def validate_persistence_config(cfg: PersistenceConfig) -> None:
    """Validate a PersistenceConfig."""
    validate_instance_config(cfg.instance)
    if not is_model_persister_target(cfg.instance._target_):
        raise ValueError(f"PersistenceConfig.instance._target_ must be a persister target, got {cfg.instance._target_}")
    validate_optional_name(cfg.name, "PersistenceConfig")


# ============================================================================
# Generative Process Config
# ============================================================================


@dataclass
class GenerativeProcessConfig:
    """Base configuration for generative processes."""

    instance: InstanceConfig
    name: str | None = None
    base_vocab_size: int = MISSING
    bos_token: int | None = MISSING
    eos_token: int | None = MISSING
    vocab_size: int = MISSING


def is_generative_process_target(target: str) -> bool:
    """Check if the target is a generative process target."""
    return target.startswith("simplexity.generative_processes.")


def is_generative_process_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generative process config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generative_process_target(target)
    return False


def validate_generative_process_config(cfg: GenerativeProcessConfig) -> None:
    """Validate a GenerativeProcessConfig."""
    validate_instance_config(cfg.instance)
    if not is_generative_process_target(cfg.instance._target_):
        raise ValueError(
            f"GenerativeProcessConfig.instance._target_ must be a generative process target, "
            f"got {cfg.instance._target_}"
        )
    validate_optional_name(cfg.name, "GenerativeProcessConfig")

    # Validate base_vocab_size
    if not isinstance(cfg.base_vocab_size, int):
        raise ValueError(f"GenerativeProcessConfig.base_vocab_size must be an int, got {type(cfg.base_vocab_size)}")
    if cfg.base_vocab_size <= 0:
        raise ValueError(f"GenerativeProcessConfig.base_vocab_size must be positive, got {cfg.base_vocab_size}")

    # Validate token values
    if cfg.bos_token is not None:
        if not isinstance(cfg.bos_token, int):
            raise ValueError(f"GenerativeProcessConfig.bos_token must be an int or None, got {type(cfg.bos_token)}")
        if cfg.bos_token < 0:
            raise ValueError(f"GenerativeProcessConfig.bos_token must be non-negative, got {cfg.bos_token}")
        if cfg.bos_token >= cfg.vocab_size:
            raise ValueError(
                f"GenerativeProcessConfig.bos_token ({cfg.bos_token}) must be < vocab_size ({cfg.vocab_size})"
            )

    if cfg.eos_token is not None:
        if not isinstance(cfg.eos_token, int):
            raise ValueError(f"GenerativeProcessConfig.eos_token must be an int or None, got {type(cfg.eos_token)}")
        if cfg.eos_token < 0:
            raise ValueError(f"GenerativeProcessConfig.eos_token must be non-negative, got {cfg.eos_token}")
        if cfg.eos_token >= cfg.vocab_size:
            raise ValueError(
                f"GenerativeProcessConfig.eos_token ({cfg.eos_token}) must be < vocab_size ({cfg.vocab_size})"
            )

    # Ensure tokens are distinct if both are set
    if cfg.bos_token is not None and cfg.eos_token is not None and cfg.bos_token == cfg.eos_token:
        raise ValueError(f"GenerativeProcessConfig.bos_token and eos_token cannot be the same ({cfg.bos_token})")

    # Validate vocab_size (should be set after resolution)
    if not isinstance(cfg.vocab_size, int):
        raise ValueError(f"GenerativeProcessConfig.vocab_size must be an int, got {type(cfg.vocab_size)}")
    if cfg.vocab_size != cfg.base_vocab_size + (cfg.bos_token is not None) + (cfg.eos_token is not None):
        raise ValueError(
            f"GenerativeProcessConfig.vocab_size ({cfg.vocab_size}) must be equal to"
            f"base_vocab_size ({cfg.base_vocab_size})"
            f"+ (bos_token is not None) ({cfg.bos_token is not None})"
            f"+ (eos_token is not None) ({cfg.eos_token is not None})"
            f"= {cfg.base_vocab_size + (cfg.bos_token is not None) + (cfg.eos_token is not None)}"
        )


# ============================================================================
# Predictive Model Configs
# ============================================================================


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


def validate_hooked_transformer_config_config(cfg: HookedTransformerConfigConfig) -> None:
    """Validate a HookedTransformerConfigConfig."""
    if cfg.d_model <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_model must be positive, got {cfg.d_model}")
    if cfg.d_head <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_head must be positive, got {cfg.d_head}")
    if cfg.n_heads <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_heads must be positive, got {cfg.n_heads}")
    if cfg.n_layers <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_layers must be positive, got {cfg.n_layers}")
    if cfg.n_ctx <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_ctx must be positive, got {cfg.n_ctx}")
    if cfg.d_mlp <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_mlp must be positive, got {cfg.d_mlp}")
    if cfg.d_vocab <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_vocab must be positive, got {cfg.d_vocab}")
    # Validate d_model is divisible by n_heads
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(
            f"HookedTransformerConfigConfig.d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )
    # Validate d_head * n_heads == d_model
    if cfg.d_head * cfg.n_heads != cfg.d_model:
        raise ValueError(
            f"HookedTransformerConfigConfig.d_head ({cfg.d_head}) * n_heads ({cfg.n_heads}) "
            f"must equal d_model ({cfg.d_model})"
        )


@dataclass
class HookedTransformerConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig


def validate_hooked_transformer_config(cfg: HookedTransformerConfig) -> None:
    """Validate a HookedTransformerConfig."""
    validate_instance_config(cfg)
    validate_hooked_transformer_config_config(cfg.cfg)


@dataclass
class ModelConfig:
    """Base configuration for predictive models."""

    instance: InstanceConfig
    name: str | None = None
    load_checkpoint_step: int | None = None


def is_predictive_model_target(target: str) -> bool:
    """Check if the target is a predictive model target."""
    parts = target.split(".")
    if len(parts) > 2:
        if parts[1] == "nn":  # torch.nn, equinox.nn, penzai.nn
            return True
        if "models" in parts[1]:  # penzai.models, simplexity.predictive_models
            return True
    return parts[0] == "transformer_lens"


def is_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_predictive_model_target(target)
    return False


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate the configuration."""
    validate_instance_config(cfg.instance)
    if not is_predictive_model_target(cfg.instance._target_):
        raise ValueError(
            f"ModelConfig.instance._target_ must be a predictive model target, got {cfg.instance._target_}"
        )
    # If this is a HookedTransformerConfig, validate it fully if we have access to the nested cfg
    if (
        cfg.instance._target_ == "transformer_lens.HookedTransformer"
        and hasattr(cfg.instance, "cfg")
        and isinstance(cfg.instance, HookedTransformerConfig)
    ):
        validate_hooked_transformer_config(cfg.instance)
    validate_optional_name(cfg.name, "ModelConfig")
    if cfg.load_checkpoint_step is not None:
        if not isinstance(cfg.load_checkpoint_step, int):
            raise ValueError(
                f"ModelConfig.load_checkpoint_step must be an int or None, got {type(cfg.load_checkpoint_step)}"
            )
        if cfg.load_checkpoint_step < 0:
            raise ValueError(f"ModelConfig.load_checkpoint_step must be non-negative, got {cfg.load_checkpoint_step}")


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
    if not isinstance(cfg.sequence_len, int):
        raise ValueError(f"TrainingConfig.sequence_len must be an int, got {type(cfg.sequence_len)}")
    if cfg.sequence_len <= 0:
        raise ValueError(f"TrainingConfig.sequence_len must be positive, got {cfg.sequence_len}")

    if not isinstance(cfg.batch_size, int):
        raise ValueError(f"TrainingConfig.batch_size must be an int, got {type(cfg.batch_size)}")
    if cfg.batch_size <= 0:
        raise ValueError(f"TrainingConfig.batch_size must be positive, got {cfg.batch_size}")

    if not isinstance(cfg.num_steps, int):
        raise ValueError(f"TrainingConfig.num_steps must be an int, got {type(cfg.num_steps)}")
    if cfg.num_steps <= 0:
        raise ValueError(f"TrainingConfig.num_steps must be positive, got {cfg.num_steps}")

    if cfg.log_every is not None:
        if not isinstance(cfg.log_every, int):
            raise ValueError(f"TrainingConfig.log_every must be an int or None, got {type(cfg.log_every)}")
        if cfg.log_every <= 0:
            raise ValueError(f"TrainingConfig.log_every must be positive, got {cfg.log_every}")
        if cfg.log_every > cfg.num_steps:
            raise ValueError(f"TrainingConfig.log_every ({cfg.log_every}) must be <= num_steps ({cfg.num_steps})")

    if cfg.validate_every is not None:
        if not isinstance(cfg.validate_every, int):
            raise ValueError(f"TrainingConfig.validate_every must be an int or None, got {type(cfg.validate_every)}")
        if cfg.validate_every <= 0:
            raise ValueError(f"TrainingConfig.validate_every must be positive, got {cfg.validate_every}")
        if cfg.validate_every > cfg.num_steps:
            raise ValueError(
                f"TrainingConfig.validate_every ({cfg.validate_every}) must be <= num_steps ({cfg.num_steps})"
            )

    if cfg.checkpoint_every is not None:
        if not isinstance(cfg.checkpoint_every, int):
            raise ValueError(
                f"TrainingConfig.checkpoint_every must be an int or None, got {type(cfg.checkpoint_every)}"
            )
        if cfg.checkpoint_every <= 0:
            raise ValueError(f"TrainingConfig.checkpoint_every must be positive, got {cfg.checkpoint_every}")
        if cfg.checkpoint_every > cfg.num_steps:
            raise ValueError(
                f"TrainingConfig.checkpoint_every ({cfg.checkpoint_every}) must be <= num_steps ({cfg.num_steps})"
            )

    # Validate optimizer config
    validate_optimizer_config(cfg.optimizer)


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
    if not isinstance(cfg.sequence_len, int):
        raise ValueError(f"ValidationConfig.sequence_len must be an int, got {type(cfg.sequence_len)}")
    if cfg.sequence_len <= 0:
        raise ValueError(f"ValidationConfig.sequence_len must be positive, got {cfg.sequence_len}")

    if not isinstance(cfg.batch_size, int):
        raise ValueError(f"ValidationConfig.batch_size must be an int, got {type(cfg.batch_size)}")
    if cfg.batch_size <= 0:
        raise ValueError(f"ValidationConfig.batch_size must be positive, got {cfg.batch_size}")

    if not isinstance(cfg.num_steps, int):
        raise ValueError(f"ValidationConfig.num_steps must be an int, got {type(cfg.num_steps)}")
    if cfg.num_steps <= 0:
        raise ValueError(f"ValidationConfig.num_steps must be positive, got {cfg.num_steps}")

    if cfg.log_every is not None:
        if not isinstance(cfg.log_every, int):
            raise ValueError(f"ValidationConfig.log_every must be an int or None, got {type(cfg.log_every)}")
        if cfg.log_every <= 0:
            raise ValueError(f"ValidationConfig.log_every must be positive, got {cfg.log_every}")
        if cfg.log_every > cfg.num_steps:
            raise ValueError(f"ValidationConfig.log_every ({cfg.log_every}) must be <= num_steps ({cfg.num_steps})")


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


def _validate_uri(uri: str, field_name: str) -> None:
    """Validate that a string is a valid URI."""
    if not isinstance(uri, str):
        raise ValueError(f"{field_name} must be a string, got {type(uri)}")
    if not uri.strip():
        raise ValueError(f"{field_name} cannot be empty")
    try:
        parsed = urlparse(uri)
        # Allow file://, http://, https://, databricks://, etc.
        if not parsed.scheme:
            raise ValueError(f"{field_name} must have a valid URI scheme (e.g., file://, http://, https://)")
    except Exception as e:
        raise ValueError(f"{field_name} is not a valid URI: {e}") from e


def validate_mlflow_config(cfg: MLFlowConfig) -> None:
    """Validate an MLFlowConfig."""
    if not isinstance(cfg.experiment_name, str) or not cfg.experiment_name.strip():
        raise ValueError("MLFlowConfig.experiment_name must be a non-empty string")
    if not isinstance(cfg.run_name, str) or not cfg.run_name.strip():
        raise ValueError("MLFlowConfig.run_name must be a non-empty string")
    if not isinstance(cfg.downgrade_unity_catalog, bool):
        raise ValueError(
            f"MLFlowConfig.downgrade_unity_catalog must be a bool, got {type(cfg.downgrade_unity_catalog)}"
        )
    _validate_uri(cfg.tracking_uri, "MLFlowConfig.tracking_uri")
    _validate_uri(cfg.registry_uri, "MLFlowConfig.registry_uri")


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
    """Validate the configuration with comprehensive checks."""
    # Validate top-level fields
    if not isinstance(cfg.seed, int):
        raise ValueError(f"MainConfig.seed must be an int, got {type(cfg.seed)}")
    if not isinstance(cfg.experiment_name, str) or not cfg.experiment_name.strip():
        raise ValueError("MainConfig.experiment_name must be a non-empty string")
    if not isinstance(cfg.run_name, str) or not cfg.run_name.strip():
        raise ValueError("MainConfig.run_name must be a non-empty string")

    # Validate sub-configs
    validate_generative_process_config(cfg.training_data_generator)
    validate_model_config(cfg.predictive_model)
    validate_training_config(cfg.training)

    # Validate optional components
    if cfg.logging is not None:
        validate_logging_config(cfg.logging)
    if cfg.persistence is not None:
        validate_persistence_config(cfg.persistence)

    # Validate validation configuration
    if validation_required(cfg):
        if cfg.validation is None:
            raise ValueError("Validation is required but not configured")
        validate_validation_config(cfg.validation)
        if cfg.validation_data_generator is None:
            raise ValueError("Validation data generator is required but not configured")
        validate_generative_process_config(cfg.validation_data_generator)
    else:
        if cfg.validation is not None:
            raise ValueError("Validation is configured but not required (validate_every is None or invalid)")
        if cfg.validation_data_generator is not None:
            raise ValueError("Validation data generator is configured but not required")

    # Validate persistence requirement
    if persistence_required(cfg):
        if cfg.persistence is None:
            raise ValueError("Persistence is required but not configured")
    else:
        if cfg.persistence is not None:
            raise ValueError("Persistence is configured but not required")

    # Validate logging requirement
    if cfg.logging is not None:
        if not logging_required(cfg):
            raise ValueError("Logging is configured but not required")
    else:
        if logging_required(cfg):
            raise ValueError("Logging is required but not configured")

    # Cross-config validation: vocab size consistency
    training_vocab_size = cfg.training_data_generator.vocab_size
    if cfg.validation_data_generator is not None:
        validation_vocab_size = cfg.validation_data_generator.vocab_size
        if training_vocab_size != validation_vocab_size:
            raise ValueError(
                f"Vocab size mismatch: training_data_generator.vocab_size ({training_vocab_size}) "
                f"!= validation_data_generator.vocab_size ({validation_vocab_size})"
            )

    # Cross-config validation: sequence length consistency
    training_seq_len = cfg.training.sequence_len
    if cfg.validation is not None:
        validation_seq_len = cfg.validation.sequence_len
        if training_seq_len != validation_seq_len:
            # This is a warning, not an error, but we'll validate it
            # Some models might support different sequence lengths, but it's unusual
            pass  # Could add a warning here if needed

    # Cross-config validation: seed consistency (optional, but worth checking)
    if cfg.training.seed != cfg.seed:
        # This is allowed but might be intentional
        pass  # Could add a warning here if needed
