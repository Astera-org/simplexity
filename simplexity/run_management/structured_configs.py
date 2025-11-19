"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError, DeviceResolutionError
from simplexity.utils.config_utils import dynamic_resolve
from simplexity.utils.pytorch_utils import resolve_device

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")

# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_nonempty_str(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a non-empty string."""
    if is_none_allowed and value is None:
        return
    if not isinstance(value, str):
        allowed_types = "a string or None" if is_none_allowed else "a string"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if not value.strip():
        raise ConfigValidationError(f"{field_name} must be a non-empty string")


def _validate_positive_int(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a positive integer."""
    if is_none_allowed and value is None:
        return
    if isinstance(value, bool):
        allowed_types = "an int or None" if is_none_allowed else "an int"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if not isinstance(value, int):
        allowed_types = "an int or None" if is_none_allowed else "an int"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if value <= 0:
        raise ConfigValidationError(f"{field_name} must be positive, got {value}")


def _validate_non_negative_int(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a non-negative integer."""
    if is_none_allowed and value is None:
        return
    if isinstance(value, bool):
        allowed_types = "an int or None" if is_none_allowed else "an int"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if not isinstance(value, int):
        allowed_types = "an int or None" if is_none_allowed else "an int"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if value < 0:
        raise ConfigValidationError(f"{field_name} must be non-negative, got {value}")


# ============================================================================
# Instance Config
# ============================================================================


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str


def _validate_instance_config(cfg: DictConfig) -> None:
    """Validate an InstanceConfig.

    Args:
        cfg: A DictConfig with an _target_ field (from Hydra).
    """
    target = cfg.get("_target_", None)
    if not isinstance(target, str):
        raise ConfigValidationError(f"InstanceConfig._target_ must be a string, got {type(target)}")
    if not target.strip():
        raise ConfigValidationError("InstanceConfig._target_ cannot be empty or whitespace")


# ============================================================================
# MLflow Config
# ============================================================================


@dataclass
class MLFlowConfig:
    """Configuration for MLflow."""

    experiment_name: str
    run_name: str
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool | None = None


def _validate_uri(uri: str, field_name: str) -> None:
    """Validate that a string is a valid URI."""
    if not uri.strip():
        raise ConfigValidationError(f"{field_name} cannot be empty")
    if uri.startswith("databricks"):
        return
    try:
        parsed = urlparse(uri)
        # Allow file://, http://, https://, databricks://, etc.
        if not parsed.scheme:
            raise ConfigValidationError(f"{field_name} must have a valid URI scheme (e.g., file://, http://, https://)")
    except Exception as e:
        raise ConfigValidationError(f"{field_name} is not a valid URI: {e}") from e


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
        raise ConfigValidationError("MLFlowConfig.experiment_name must be a non-empty string")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ConfigValidationError("MLFlowConfig.run_name must be a non-empty string")
    if downgrade_unity_catalog is not None and not isinstance(downgrade_unity_catalog, bool):
        raise ConfigValidationError(
            f"MLFlowConfig.downgrade_unity_catalog must be a bool, got {type(downgrade_unity_catalog)}"
        )
    if tracking_uri is not None:
        _validate_uri(tracking_uri, "MLFlowConfig.tracking_uri")
    if registry_uri is not None:
        _validate_uri(registry_uri, "MLFlowConfig.registry_uri")


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
        raise ConfigValidationError("LoggingConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_logger_target(target):
        raise ConfigValidationError(f"LoggingConfig.instance._target_ must be a logger target, got {target}")
    _validate_nonempty_str(cfg.get("name"), "LoggingConfig.name", is_none_allowed=True)


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
    sequence_len: int | None = MISSING
    batch_size: int | None = None


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
        raise ConfigValidationError("GenerativeProcessConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_generative_process_target(target):
        raise ConfigValidationError(
            f"GenerativeProcessConfig.instance._target_ must be a generative process target, got {target}"
        )
    _validate_nonempty_str(cfg.get("name"), "GenerativeProcessConfig.name", is_none_allowed=True)

    base_vocab_size = cfg.get("base_vocab_size")
    if OmegaConf.is_missing(cfg, "base_vocab_size"):
        SIMPLEXITY_LOGGER.debug("[generative process] base_vocab_size is missing, will be resolved dynamically")
    else:
        _validate_positive_int(base_vocab_size, "GenerativeProcessConfig.base_vocab_size")

    # Validate token values
    bos_token = cfg.get("bos_token")
    eos_token = cfg.get("eos_token")
    vocab_size = cfg.get("vocab_size")

    if not OmegaConf.is_missing(cfg, "vocab_size"):
        _validate_positive_int(vocab_size, "GenerativeProcessConfig.vocab_size")

    if OmegaConf.is_missing(cfg, "bos_token"):
        SIMPLEXITY_LOGGER.debug("[generative process] bos_token is missing, will be resolved dynamically")
    elif bos_token is not None:
        _validate_non_negative_int(bos_token, "GenerativeProcessConfig.bos_token", is_none_allowed=True)
        if not OmegaConf.is_missing(cfg, "vocab_size") and bos_token >= vocab_size:
            raise ConfigValidationError(
                f"GenerativeProcessConfig.bos_token ({bos_token}) must be < vocab_size ({vocab_size})"
            )

    if OmegaConf.is_missing(cfg, "eos_token"):
        SIMPLEXITY_LOGGER.debug("[generative process] eos_token is missing, will be resolved dynamically")
    elif eos_token is not None:
        _validate_non_negative_int(eos_token, "GenerativeProcessConfig.eos_token", is_none_allowed=True)
        if not OmegaConf.is_missing(cfg, "vocab_size") and eos_token >= vocab_size:
            raise ConfigValidationError(
                f"GenerativeProcessConfig.eos_token ({eos_token}) must be < vocab_size ({vocab_size})"
            )

    # Ensure tokens are distinct if both are set (skip if either is MISSING)
    if (
        not OmegaConf.is_missing(cfg, "bos_token")
        and not OmegaConf.is_missing(cfg, "eos_token")
        and bos_token is not None
        and eos_token is not None
        and bos_token == eos_token
    ):
        raise ConfigValidationError(f"GenerativeProcessConfig.bos_token and eos_token cannot be the same ({bos_token})")

    if OmegaConf.is_missing(cfg, "vocab_size"):
        SIMPLEXITY_LOGGER.debug("[generative process] vocab_size is missing, will be resolved dynamically")
    else:
        # Only validate consistency if base_vocab_size is also resolved
        if not OmegaConf.is_missing(cfg, "base_vocab_size"):
            _validate_positive_int(base_vocab_size, "GenerativeProcessConfig.base_vocab_size")
            use_bos_token = bos_token is not None or OmegaConf.is_missing(cfg, "bos_token")
            use_eos_token = eos_token is not None or OmegaConf.is_missing(cfg, "eos_token")
            expected_vocab_size = base_vocab_size + use_bos_token + use_eos_token
            if vocab_size != expected_vocab_size:
                raise ConfigValidationError(
                    f"GenerativeProcessConfig.vocab_size ({vocab_size}) must be equal to "
                    f"base_vocab_size ({base_vocab_size}) "
                    f"+ use_bos_token ({use_bos_token}) "
                    f"+ use_eos_token ({use_eos_token}) "
                    f"= {expected_vocab_size}"
                )

    sequence_len = cfg.get("sequence_len")
    if OmegaConf.is_missing(cfg, "sequence_len"):
        SIMPLEXITY_LOGGER.debug("[generative process] sequence_len is missing, will be resolved dynamically")
    else:
        _validate_positive_int(sequence_len, "GenerativeProcessConfig.sequence_len", is_none_allowed=True)

    batch_size = cfg.get("batch_size")
    if OmegaConf.is_missing(cfg, "batch_size"):
        SIMPLEXITY_LOGGER.debug("[generative process] batch_size is missing, will be resolved dynamically")
    else:
        _validate_positive_int(batch_size, "GenerativeProcessConfig.batch_size", is_none_allowed=True)


@dynamic_resolve
def resolve_generative_process_config(cfg: DictConfig, base_vocab_size: int) -> None:
    """Resolve the GenerativeProcessConfig."""
    # Resolve base_vocab_size
    if OmegaConf.is_missing(cfg, "base_vocab_size"):
        cfg.base_vocab_size = base_vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] base_vocab_size resolved to: %s", base_vocab_size)
    elif cfg.get("base_vocab_size") != base_vocab_size:
        raise ConfigValidationError(
            f"GenerativeProcessConfig.base_vocab_size ({cfg.get('base_vocab_size')}) must be equal to {base_vocab_size}"
        )
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] base_vocab_size defined as: %s", cfg.get("base_vocab_size"))
    vocab_size = base_vocab_size
    # Resolve bos_token
    if OmegaConf.is_missing(cfg, "bos_token"):
        cfg.bos_token = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] bos_token resolved to: %s", cfg.bos_token)
        vocab_size += 1
    elif cfg.get("bos_token", None) is not None:
        bos_token = cfg.get("bos_token")
        SIMPLEXITY_LOGGER.debug("[generative process] bos_token defined as: %s", bos_token)
        vocab_size = max(vocab_size, bos_token + 1)
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] no bos_token set")
    # Resolve eos_token
    if OmegaConf.is_missing(cfg, "eos_token"):
        cfg.eos_token = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] eos_token resolved to: %s", cfg.eos_token)
        vocab_size += 1
    elif cfg.get("eos_token", None) is not None:
        eos_token = cfg.get("eos_token")
        SIMPLEXITY_LOGGER.debug("[generative process] eos_token defined as: %s", eos_token)
        vocab_size = max(vocab_size, eos_token + 1)
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] no eos_token set")
    # Resolve vocab_size
    if OmegaConf.is_missing(cfg, "vocab_size"):
        cfg.vocab_size = vocab_size
        SIMPLEXITY_LOGGER.info("[generative process] vocab_size resolved to: %s", vocab_size)
    elif cfg.get("vocab_size") != vocab_size:
        raise ConfigValidationError(
            f"GenerativeProcessConfig.vocab_size ({cfg.get('vocab_size')}) must be equal to {vocab_size}"
        )
    else:
        SIMPLEXITY_LOGGER.debug("[generative process] vocab_size defined as: %s", cfg.get("vocab_size"))


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
        raise ConfigValidationError("PersistenceConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_model_persister_target(target):
        raise ConfigValidationError(f"PersistenceConfig.instance._target_ must be a persister target, got {target}")
    _validate_nonempty_str(cfg.get("name"), "PersistenceConfig.name", is_none_allowed=True)


# ============================================================================
# Predictive Model Configs
# ============================================================================


@dataclass
class HookedTransformerConfigConfig:
    """Configuration for HookedTransformerConfig."""

    n_layers: int
    d_model: int
    d_head: int
    n_ctx: int = MISSING
    n_heads: int = -1
    d_mlp: int | None = None
    act_fn: str | None = None
    d_vocab: int = MISSING
    normalization_type: str | None = "LN"
    device: str | None = None
    seed: int | None = None
    _target_: str = "transformer_lens.HookedTransformerConfig"


def validate_hooked_transformer_config_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerConfigConfig.

    Args:
        cfg: A DictConfig with HookedTransformerConfigConfig fields (from Hydra).
    """
    n_layers = cfg.get("n_layers")
    d_model = cfg.get("d_model")
    d_head = cfg.get("d_head")
    n_ctx = cfg.get("n_ctx")
    n_heads = cfg.get("n_heads")
    d_mlp = cfg.get("d_mlp")
    d_vocab = cfg.get("d_vocab")
    seed = cfg.get("seed")

    _validate_positive_int(n_layers, "HookedTransformerConfigConfig.n_layers")
    _validate_positive_int(d_model, "HookedTransformerConfigConfig.d_model")
    _validate_positive_int(d_head, "HookedTransformerConfigConfig.d_head")
    if OmegaConf.is_missing(cfg, "n_ctx"):
        SIMPLEXITY_LOGGER.debug("[predictive model] n_ctx is missing, will be resolved dynamically")
    else:
        _validate_positive_int(n_ctx, "HookedTransformerConfigConfig.n_ctx")
    if n_heads != -1:
        _validate_positive_int(n_heads, "HookedTransformerConfigConfig.n_heads")
        if d_model % n_heads != 0:
            raise ConfigValidationError(
                f"HookedTransformerConfigConfig.d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if d_head * n_heads != d_model:
            raise ConfigValidationError(
                f"HookedTransformerConfigConfig.d_head ({d_head}) * n_heads ({n_heads}) must equal d_model ({d_model})"
            )
    elif d_model % d_head != 0:
        raise ConfigValidationError(
            f"HookedTransformerConfigConfig.d_model ({d_model}) must be divisible by d_head ({d_head})"
        )

    _validate_positive_int(d_mlp, "HookedTransformerConfigConfig.d_mlp", is_none_allowed=True)
    if OmegaConf.is_missing(cfg, "d_vocab"):
        SIMPLEXITY_LOGGER.debug("[predictive model] d_vocab is missing, will be resolved dynamically")
    else:
        _validate_positive_int(d_vocab, "HookedTransformerConfigConfig.d_vocab")
    _validate_non_negative_int(seed, "HookedTransformerConfigConfig.seed", is_none_allowed=True)


@dataclass
class HookedTransformerConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig

    def __init__(self, cfg: HookedTransformerConfigConfig, _target_: str = "transformer_lens.HookedTransformer"):
        super().__init__(_target_=_target_)
        self.cfg = cfg


def validate_hooked_transformer_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerConfig.

    Args:
        cfg: A DictConfig with _target_ and cfg fields (from Hydra).
    """
    _validate_instance_config(cfg)
    nested_cfg = cfg.get("cfg")
    if nested_cfg is None:
        raise ConfigValidationError("HookedTransformerConfig.cfg is required")
    validate_hooked_transformer_config_config(nested_cfg)


@dynamic_resolve
def resolve_hooked_transformer_config(
    cfg: DictConfig,
    *,
    vocab_size: int | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
    sequence_len: int | None = None,
) -> None:
    """Resolve the HookedTransformerConfig."""
    # Resolve d_vocab
    if vocab_size is None:
        SIMPLEXITY_LOGGER.debug("[predictive model] no vocab_size set")
    else:
        if OmegaConf.is_missing(cfg, "d_vocab"):
            cfg.d_vocab = vocab_size
            SIMPLEXITY_LOGGER.info("[predictive model] d_vocab resolved to: %s", vocab_size)
        elif cfg.get("d_vocab") != vocab_size:
            raise ConfigValidationError(
                f"HookedTransformerConfig.d_vocab ({cfg.get('d_vocab')}) must be equal to {vocab_size}"
            )
        else:
            SIMPLEXITY_LOGGER.debug("[predictive model] d_vocab defined as: %s", cfg.get("d_vocab"))
    # Resolve n_ctx
    if sequence_len is None:
        SIMPLEXITY_LOGGER.debug("[predictive model] no sequence_len set")
    else:
        use_bos = bos_token is not None
        use_eos = eos_token is not None
        n_ctx = sequence_len + int(use_bos) + int(use_eos) - 1
        if OmegaConf.is_missing(cfg, "n_ctx"):
            cfg.n_ctx = n_ctx
            SIMPLEXITY_LOGGER.info("[predictive model] n_ctx resolved to: %s", n_ctx)
        elif cfg.get("n_ctx") != n_ctx:
            raise ConfigValidationError(f"HookedTransformerConfig.n_ctx ({cfg.get('n_ctx')}) must be equal to {n_ctx}")
        else:
            SIMPLEXITY_LOGGER.debug("[predictive model] n_ctx defined as: %s", cfg.get("n_ctx"))
    # Resolve device
    device: str | None = cfg.get("device", None)
    try:
        resolved_device = resolve_device(device)
    except DeviceResolutionError as e:
        SIMPLEXITY_LOGGER.warning("[predictive model] specified device %s could not be resolved: %s", device, e)
        resolved_device = "cpu"
    if device is None or device == "auto":
        cfg.device = resolved_device
        SIMPLEXITY_LOGGER.info("[predictive model] device resolved to: %s", resolved_device)
    elif device != resolved_device:
        cfg.device = resolved_device
        SIMPLEXITY_LOGGER.warning("[predictive model] specified device %s resolved to %s", device, resolved_device)
    else:
        SIMPLEXITY_LOGGER.debug("[predictive model] device defined as: %s", device)


@dataclass
class PredictiveModelConfig:
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
        if "models" in parts[1]:  # penzai.models
            return True
    return parts[0] == "transformer_lens"


def is_predictive_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_predictive_model_target(target)
    return False


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"


def validate_predictive_model_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with instance, optional name, and optional load_checkpoint_step fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("PredictiveModelConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_predictive_model_target(target):
        raise ConfigValidationError(
            f"PredictiveModelConfig.instance._target_ must be a predictive model target, got {target}"
        )
    if target == "transformer_lens.HookedTransformer" and instance.get("cfg") is not None:
        validate_hooked_transformer_config(instance)
    _validate_nonempty_str(cfg.get("name"), "PredictiveModelConfig.name", is_none_allowed=True)
    load_checkpoint_step = cfg.get("load_checkpoint_step")
    if load_checkpoint_step is not None:
        _validate_non_negative_int(
            load_checkpoint_step, "PredictiveModelConfig.load_checkpoint_step", is_none_allowed=True
        )


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
        raise ConfigValidationError("OptimizerConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_optimizer_target(target):
        raise ConfigValidationError(f"OptimizerConfig.instance._target_ must be an optimizer target, got {target}")
    _validate_nonempty_str(cfg.get("name"), "OptimizerConfig.name", is_none_allowed=True)


# ============================================================================
# Activation Analysis Config
# ============================================================================


@dataclass
class ActivationAnalysisConfig:
    """Base configuration for activation analyses."""

    instance: InstanceConfig
    name: str | None = None


@dataclass
class ActivationTrackerConfig:
    """Configuration for activation tracker."""

    analyses: dict[str, ActivationAnalysisConfig]
    name: str | None = None


def is_activation_analysis_target(target: str) -> bool:
    """Check if the target is an activation analysis target."""
    return target.startswith("simplexity.activations.")


def is_activation_tracker_target(target: str) -> bool:
    """Check if the target is an activation tracker target."""
    return target == "simplexity.activations.activation_tracker.ActivationTracker"


def validate_activation_analysis_config(cfg: DictConfig) -> None:
    """Validate an ActivationAnalysisConfig.
    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("ActivationAnalysisConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_activation_analysis_target(target):
        raise ConfigValidationError(
            f"ActivationAnalysisConfig.instance._target_ must be an activation analysis target, got {target}"
        )
    _validate_nonempty_str(cfg.get("name"), "ActivationAnalysisConfig.name", is_none_allowed=True)


def validate_activation_tracker_config(cfg: DictConfig) -> None:
    """Validate an ActivationTrackerConfig.
    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("ActivationTrackerConfig.instance is required")
    _validate_instance_config(instance)
    target = instance.get("_target_", None)
    if not is_activation_tracker_target(target):
        raise ConfigValidationError(
            f"ActivationTrackerConfig.instance._target_ must be ActivationTracker, got {target}"
        )

    analyses = instance.get("analyses")
    if analyses is None:
        raise ConfigValidationError("ActivationTrackerConfig.instance.analyses is required")

    if not isinstance(analyses, DictConfig):
        raise ConfigValidationError("ActivationTrackerConfig.instance.analyses must be a dictionary")

    for key, analysis_config in analyses.items():
        if not isinstance(analysis_config, DictConfig):
            raise ConfigValidationError(f"ActivationTrackerConfig.instance.analyses[{key}] must be a config dict")
        target = analysis_config.get("_target_", None)
        if not is_activation_analysis_target(target):
            raise ConfigValidationError(
                f"ActivationTrackerConfig.instance.analyses[{key}]._target_"
                f" must be an activation analysis, got {target}"
            )

    _validate_nonempty_str(cfg.get("name"), "ActivationTrackerConfig.name", is_none_allowed=True)


# ============================================================================
# Base Config
# ============================================================================


@dataclass
class BaseConfig:
    """Base configuration for all components."""

    seed: int | None = None
    tags: dict[str, str] | None = None
    mlflow: MLFlowConfig | None = None


def validate_base_config(cfg: DictConfig) -> None:
    """Validate a BaseConfig.

    Args:
        cfg: A DictConfig with seed, tags, and mlflow fields (from Hydra).
    """
    seed = cfg.get("seed")
    _validate_non_negative_int(seed, "BaseConfig.seed", is_none_allowed=True)
    tags = cfg.get("tags")
    if tags is not None:
        if not isinstance(tags, DictConfig):
            raise ConfigValidationError("BaseConfig.tags must be a dictionary")
        for key, value in tags.items():
            if not isinstance(key, str):
                raise ConfigValidationError(f"BaseConfig.tags keys must be strings, got {type(key)}")
            if not isinstance(value, str):
                raise ConfigValidationError(f"BaseConfig.tags values must be strings, got {type(value)}")
    mlflow = cfg.get("mlflow")
    if mlflow is not None:
        if not isinstance(mlflow, DictConfig):
            raise ConfigValidationError("BaseConfig.mlflow must be a MLFlowConfig")
        validate_mlflow_config(mlflow)
