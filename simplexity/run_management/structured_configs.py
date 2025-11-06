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


def validate_instance_config(cfg: DictConfig) -> None:
    """Validate an InstanceConfig.

    Args:
        cfg: A DictConfig with an _target_ field (from Hydra).
    """
    target = cfg.get("_target_", None)
    if not isinstance(target, str):
        raise ValueError(f"InstanceConfig._target_ must be a string, got {type(target)}")
    if not target.strip():
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


def validate_optimizer_config(cfg: DictConfig) -> None:
    """Validate an OptimizerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ValueError("OptimizerConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_optimizer_target(target):
        raise ValueError(f"OptimizerConfig.instance._target_ must be an optimizer target, got {target}")
    validate_optional_name(cfg.get("name"), "OptimizerConfig")


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


def validate_logging_config(cfg: DictConfig) -> None:
    """Validate a LoggingConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ValueError("LoggingConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_logger_target(target):
        raise ValueError(f"LoggingConfig.instance._target_ must be a logger target, got {target}")
    validate_optional_name(cfg.get("name"), "LoggingConfig")


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


def validate_persistence_config(cfg: DictConfig) -> None:
    """Validate a PersistenceConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ValueError("PersistenceConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_model_persister_target(target):
        raise ValueError(f"PersistenceConfig.instance._target_ must be a persister target, got {target}")
    validate_optional_name(cfg.get("name"), "PersistenceConfig")


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


def validate_generative_process_config(cfg: DictConfig) -> None:
    """Validate a GenerativeProcessConfig.

    Args:
        cfg: A DictConfig with instance, name, base_vocab_size, bos_token, eos_token,
             and vocab_size fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ValueError("GenerativeProcessConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_generative_process_target(target):
        raise ValueError(f"GenerativeProcessConfig.instance._target_ must be a generative process target, got {target}")
    validate_optional_name(cfg.get("name"), "GenerativeProcessConfig")

    # Validate base_vocab_size
    base_vocab_size = cfg.get("base_vocab_size")
    if not isinstance(base_vocab_size, int):
        raise ValueError(f"GenerativeProcessConfig.base_vocab_size must be an int, got {type(base_vocab_size)}")
    if base_vocab_size <= 0:
        raise ValueError(f"GenerativeProcessConfig.base_vocab_size must be positive, got {base_vocab_size}")

    # Validate token values
    bos_token = cfg.get("bos_token")
    eos_token = cfg.get("eos_token")
    vocab_size = cfg.get("vocab_size")

    if bos_token is not None:
        if not isinstance(bos_token, int):
            raise ValueError(f"GenerativeProcessConfig.bos_token must be an int or None, got {type(bos_token)}")
        if bos_token < 0:
            raise ValueError(f"GenerativeProcessConfig.bos_token must be non-negative, got {bos_token}")
        if vocab_size is not None and bos_token >= vocab_size:
            raise ValueError(f"GenerativeProcessConfig.bos_token ({bos_token}) must be < vocab_size ({vocab_size})")

    if eos_token is not None:
        if not isinstance(eos_token, int):
            raise ValueError(f"GenerativeProcessConfig.eos_token must be an int or None, got {type(eos_token)}")
        if eos_token < 0:
            raise ValueError(f"GenerativeProcessConfig.eos_token must be non-negative, got {eos_token}")
        if vocab_size is not None and eos_token >= vocab_size:
            raise ValueError(f"GenerativeProcessConfig.eos_token ({eos_token}) must be < vocab_size ({vocab_size})")

    # Ensure tokens are distinct if both are set
    if bos_token is not None and eos_token is not None and bos_token == eos_token:
        raise ValueError(f"GenerativeProcessConfig.bos_token and eos_token cannot be the same ({bos_token})")

    # Validate vocab_size (should be set after resolution)
    if not isinstance(vocab_size, int):
        raise ValueError(f"GenerativeProcessConfig.vocab_size must be an int, got {type(vocab_size)}")
    expected_vocab_size = base_vocab_size + (bos_token is not None) + (eos_token is not None)
    if vocab_size != expected_vocab_size:
        raise ValueError(
            f"GenerativeProcessConfig.vocab_size ({vocab_size}) must be equal to "
            f"base_vocab_size ({base_vocab_size}) "
            f"+ (bos_token is not None) ({bos_token is not None}) "
            f"+ (eos_token is not None) ({eos_token is not None}) "
            f"= {expected_vocab_size}"
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


def validate_hooked_transformer_config_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerConfigConfig.

    Args:
        cfg: A DictConfig with HookedTransformerConfigConfig fields (from Hydra).
    """
    d_model = cfg.get("d_model")
    d_head = cfg.get("d_head")
    n_heads = cfg.get("n_heads")
    n_layers = cfg.get("n_layers")
    n_ctx = cfg.get("n_ctx")
    d_mlp = cfg.get("d_mlp")
    d_vocab = cfg.get("d_vocab")

    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_model must be positive, got {d_model}")
    if not isinstance(d_head, int) or d_head <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_head must be positive, got {d_head}")
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_heads must be positive, got {n_heads}")
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_layers must be positive, got {n_layers}")
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.n_ctx must be positive, got {n_ctx}")
    if not isinstance(d_mlp, int) or d_mlp <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_mlp must be positive, got {d_mlp}")
    if not isinstance(d_vocab, int) or d_vocab <= 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_vocab must be positive, got {d_vocab}")
    # Validate d_model is divisible by n_heads
    if d_model % n_heads != 0:
        raise ValueError(f"HookedTransformerConfigConfig.d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    # Validate d_head * n_heads == d_model
    if d_head * n_heads != d_model:
        raise ValueError(
            f"HookedTransformerConfigConfig.d_head ({d_head}) * n_heads ({n_heads}) must equal d_model ({d_model})"
        )


@dataclass
class HookedTransformerConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig


def validate_hooked_transformer_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerConfig.

    Args:
        cfg: A DictConfig with _target_ and cfg fields (from Hydra).
    """
    validate_instance_config(cfg)
    nested_cfg = cfg.get("cfg")
    if nested_cfg is None:
        raise ValueError("HookedTransformerConfig.cfg is required")
    validate_hooked_transformer_config_config(nested_cfg)


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


def validate_model_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with instance, optional name, and optional load_checkpoint_step fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ValueError("ModelConfig.instance is required")
    validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_predictive_model_target(target):
        raise ValueError(f"ModelConfig.instance._target_ must be a predictive model target, got {target}")
    # If this is a HookedTransformerConfig, validate it fully if we have access to the nested cfg
    if target == "transformer_lens.HookedTransformer" and instance.get("cfg") is not None:
        validate_hooked_transformer_config(instance)
    validate_optional_name(cfg.get("name"), "ModelConfig")
    load_checkpoint_step = cfg.get("load_checkpoint_step")
    if load_checkpoint_step is not None:
        if not isinstance(load_checkpoint_step, int):
            raise ValueError(
                f"ModelConfig.load_checkpoint_step must be an int or None, got {type(load_checkpoint_step)}"
            )
        if load_checkpoint_step < 0:
            raise ValueError(f"ModelConfig.load_checkpoint_step must be non-negative, got {load_checkpoint_step}")


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


def validate_training_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with TrainingConfig fields (from Hydra).
    """
    sequence_len = cfg.get("sequence_len")
    batch_size = cfg.get("batch_size")
    num_steps = cfg.get("num_steps")
    log_every = cfg.get("log_every")
    validate_every = cfg.get("validate_every")
    checkpoint_every = cfg.get("checkpoint_every")
    optimizer = cfg.get("optimizer")

    if not isinstance(sequence_len, int):
        raise ValueError(f"TrainingConfig.sequence_len must be an int, got {type(sequence_len)}")
    if sequence_len <= 0:
        raise ValueError(f"TrainingConfig.sequence_len must be positive, got {sequence_len}")

    if not isinstance(batch_size, int):
        raise ValueError(f"TrainingConfig.batch_size must be an int, got {type(batch_size)}")
    if batch_size <= 0:
        raise ValueError(f"TrainingConfig.batch_size must be positive, got {batch_size}")

    if not isinstance(num_steps, int):
        raise ValueError(f"TrainingConfig.num_steps must be an int, got {type(num_steps)}")
    if num_steps <= 0:
        raise ValueError(f"TrainingConfig.num_steps must be positive, got {num_steps}")

    if log_every is not None:
        if not isinstance(log_every, int):
            raise ValueError(f"TrainingConfig.log_every must be an int or None, got {type(log_every)}")
        if log_every <= 0:
            raise ValueError(f"TrainingConfig.log_every must be positive, got {log_every}")
        if log_every > num_steps:
            raise ValueError(f"TrainingConfig.log_every ({log_every}) must be <= num_steps ({num_steps})")

    if validate_every is not None:
        if not isinstance(validate_every, int):
            raise ValueError(f"TrainingConfig.validate_every must be an int or None, got {type(validate_every)}")
        if validate_every <= 0:
            raise ValueError(f"TrainingConfig.validate_every must be positive, got {validate_every}")
        if validate_every > num_steps:
            raise ValueError(f"TrainingConfig.validate_every ({validate_every}) must be <= num_steps ({num_steps})")

    if checkpoint_every is not None:
        if not isinstance(checkpoint_every, int):
            raise ValueError(f"TrainingConfig.checkpoint_every must be an int or None, got {type(checkpoint_every)}")
        if checkpoint_every <= 0:
            raise ValueError(f"TrainingConfig.checkpoint_every must be positive, got {checkpoint_every}")
        if checkpoint_every > num_steps:
            raise ValueError(f"TrainingConfig.checkpoint_every ({checkpoint_every}) must be <= num_steps ({num_steps})")

    # Validate optimizer config
    if optimizer is None:
        raise ValueError("TrainingConfig.optimizer is required")
    validate_optimizer_config(optimizer)


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


def validate_validation_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with ValidationConfig fields (from Hydra).
    """
    sequence_len = cfg.get("sequence_len")
    batch_size = cfg.get("batch_size")
    num_steps = cfg.get("num_steps")
    log_every = cfg.get("log_every")

    if not isinstance(sequence_len, int):
        raise ValueError(f"ValidationConfig.sequence_len must be an int, got {type(sequence_len)}")
    if sequence_len <= 0:
        raise ValueError(f"ValidationConfig.sequence_len must be positive, got {sequence_len}")

    if not isinstance(batch_size, int):
        raise ValueError(f"ValidationConfig.batch_size must be an int, got {type(batch_size)}")
    if batch_size <= 0:
        raise ValueError(f"ValidationConfig.batch_size must be positive, got {batch_size}")

    if not isinstance(num_steps, int):
        raise ValueError(f"ValidationConfig.num_steps must be an int, got {type(num_steps)}")
    if num_steps <= 0:
        raise ValueError(f"ValidationConfig.num_steps must be positive, got {num_steps}")

    if log_every is not None:
        if not isinstance(log_every, int):
            raise ValueError(f"ValidationConfig.log_every must be an int or None, got {type(log_every)}")
        if log_every <= 0:
            raise ValueError(f"ValidationConfig.log_every must be positive, got {log_every}")
        if log_every > num_steps:
            raise ValueError(f"ValidationConfig.log_every ({log_every}) must be <= num_steps ({num_steps})")


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


def validate_mlflow_config(cfg: DictConfig) -> None:
    """Validate an MLFlowConfig.

    Args:
        cfg: A DictConfig with MLFlowConfig fields (from Hydra).
    """
    experiment_name = cfg.get("experiment_name")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("MLFlowConfig.experiment_name must be a non-empty string")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("MLFlowConfig.run_name must be a non-empty string")
    if not isinstance(downgrade_unity_catalog, bool):
        raise ValueError(f"MLFlowConfig.downgrade_unity_catalog must be a bool, got {type(downgrade_unity_catalog)}")
    _validate_uri(tracking_uri, "MLFlowConfig.tracking_uri")
    _validate_uri(registry_uri, "MLFlowConfig.registry_uri")


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


def validation_required(cfg: DictConfig) -> bool:
    """Check if validation is required."""
    training = cfg.get("training")
    if training is None:
        return False
    validate_every = training.get("validate_every")
    num_steps = training.get("num_steps")
    return validate_every is not None and validate_every > 0 and num_steps is not None and validate_every <= num_steps


def persistence_required(cfg: DictConfig) -> bool:
    """Check if persistence is required."""
    predictive_model = cfg.get("predictive_model")
    training = cfg.get("training")
    if training is None:
        return False
    checkpoint_every = training.get("checkpoint_every")
    num_steps = training.get("num_steps")
    return (predictive_model is not None and predictive_model.get("load_checkpoint_step") is not None) or (
        checkpoint_every is not None
        and checkpoint_every > 0
        and num_steps is not None
        and checkpoint_every <= num_steps
    )


def logging_required(cfg: DictConfig) -> bool:
    """Check if logging is required."""
    training = cfg.get("training")
    if training is None:
        return False
    log_every = training.get("log_every")
    num_steps = training.get("num_steps")
    if log_every is not None and log_every > 0 and num_steps is not None and log_every <= num_steps:
        return True
    validation = cfg.get("validation")
    if validation is None:
        return False
    validation_log_every = validation.get("log_every")
    validation_num_steps = validation.get("num_steps")
    return bool(
        validation_required(cfg)
        and validation_log_every is not None
        and validation_log_every > 0
        and validation_num_steps is not None
        and validation_log_every <= validation_num_steps
    )


def validate_config(cfg: DictConfig) -> None:
    """Validate the configuration with comprehensive checks.

    Args:
        cfg: A DictConfig matching the MainConfig structure (from Hydra).
    """
    # Validate top-level fields
    seed = cfg.get("seed")
    experiment_name = cfg.get("experiment_name")
    run_name = cfg.get("run_name")

    if not isinstance(seed, int):
        raise ValueError(f"MainConfig.seed must be an int, got {type(seed)}")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("MainConfig.experiment_name must be a non-empty string")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("MainConfig.run_name must be a non-empty string")

    # Validate sub-configs
    training_data_generator = cfg.get("training_data_generator")
    predictive_model = cfg.get("predictive_model")
    training = cfg.get("training")
    logging = cfg.get("logging")
    persistence = cfg.get("persistence")
    validation = cfg.get("validation")
    validation_data_generator = cfg.get("validation_data_generator")

    if training_data_generator is None:
        raise ValueError("MainConfig.training_data_generator is required")
    if predictive_model is None:
        raise ValueError("MainConfig.predictive_model is required")
    if training is None:
        raise ValueError("MainConfig.training is required")

    validate_generative_process_config(training_data_generator)
    validate_model_config(predictive_model)
    validate_training_config(training)

    # Validate optional components
    if logging is not None:
        validate_logging_config(logging)
    if persistence is not None:
        validate_persistence_config(persistence)

    # Validate validation configuration
    if validation_required(cfg):
        if validation is None:
            raise ValueError("Validation is required but not configured")
        validate_validation_config(validation)
        if validation_data_generator is None:
            raise ValueError("Validation data generator is required but not configured")
        validate_generative_process_config(validation_data_generator)
    else:
        if validation is not None:
            raise ValueError("Validation is configured but not required (validate_every is None or invalid)")
        if validation_data_generator is not None:
            raise ValueError("Validation data generator is configured but not required")

    # Validate persistence requirement
    if persistence_required(cfg):
        if persistence is None:
            raise ValueError("Persistence is required but not configured")
    else:
        if persistence is not None:
            raise ValueError("Persistence is configured but not required")

    # Validate logging requirement
    if logging is not None:
        if not logging_required(cfg):
            raise ValueError("Logging is configured but not required")
    else:
        if logging_required(cfg):
            raise ValueError("Logging is required but not configured")

    # Cross-config validation: vocab size consistency
    training_vocab_size = training_data_generator.get("vocab_size")
    if validation_data_generator is not None:
        validation_vocab_size = validation_data_generator.get("vocab_size")
        if training_vocab_size != validation_vocab_size:
            raise ValueError(
                f"Vocab size mismatch: training_data_generator.vocab_size ({training_vocab_size}) "
                f"!= validation_data_generator.vocab_size ({validation_vocab_size})"
            )

    # Cross-config validation: sequence length consistency
    training_seq_len = training.get("sequence_len")
    if validation is not None:
        validation_seq_len = validation.get("sequence_len")
        if training_seq_len != validation_seq_len:
            # This is a warning, not an error, but we'll validate it
            # Some models might support different sequence lengths, but it's unusual
            pass  # Could add a warning here if needed

    # Cross-config validation: seed consistency (optional, but worth checking)
    training_seed = training.get("seed")
    if training_seed != seed:
        # This is allowed but might be intentional
        pass  # Could add a warning here if needed
