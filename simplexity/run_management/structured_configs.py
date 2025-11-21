"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import jax
import jax.numpy as jnp
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


def _validate_positive_float(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a positive float."""
    if is_none_allowed and value is None:
        return
    if not isinstance(value, float):
        allowed_types = "a float or None" if is_none_allowed else "a float"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if value <= 0:
        raise ConfigValidationError(f"{field_name} must be positive, got {value}")


def _validate_non_negative_float(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a non-negative float."""
    if is_none_allowed and value is None:
        return
    if not isinstance(value, float):
        allowed_types = "a float or None" if is_none_allowed else "a float"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if value < 0:
        raise ConfigValidationError(f"{field_name} must be non-negative, got {value}")


def _validate_bool(value: Any, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a value is a boolean."""
    if is_none_allowed and value is None:
        return
    if not isinstance(value, bool):
        allowed_types = "a bool or None" if is_none_allowed else "a bool"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")


def _validate_sequence(
    value: Any,
    field_name: str,
    element_type: type | None = None,
    is_none_allowed: bool = False,
) -> None:
    """Validate that a value is a sequence of elements of a given type."""
    if is_none_allowed and value is None:
        return
    if isinstance(value, jax.Array):
        if value.ndim != 1:
            raise ConfigValidationError(f"{field_name} must be a 1D array, got {value.shape}")
        if element_type is float and value.dtype not in [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]:
            raise ConfigValidationError(f"{field_name} must be a float array, got {value.dtype}")
        return
    if not isinstance(value, Sequence):
        allowed_types = "a sequence or None" if is_none_allowed else "a sequence"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if element_type is None:
        return
    for item in value:
        if not isinstance(item, element_type):
            raise ConfigValidationError(f"{field_name} items must be floats, got {type(item)}")


def _validate_mapping(
    value: Any,
    field_name: str,
    key_type: type | None = None,
    value_type: type | None = None,
    is_none_allowed: bool = False,
) -> None:
    """Validate that a value is a dictionary with keys of a given type and values of a given type."""
    if is_none_allowed and value is None:
        return
    if not isinstance(value, Mapping):
        allowed_types = "a dictionary or None" if is_none_allowed else "a dictionary"
        raise ConfigValidationError(f"{field_name} must be {allowed_types}, got {type(value)}")
    if key_type is not None and not all(isinstance(key, key_type) for key in value):
        raise ConfigValidationError(f"{field_name} keys must be {key_type.__name__}s")
    if value_type is not None and not all(isinstance(value, value_type) for value in value.values()):
        raise ConfigValidationError(f"{field_name} values must be {value_type.__name__}s")


def _validate_uri(uri: str, field_name: str, is_none_allowed: bool = False) -> None:
    """Validate that a string is a valid URI."""
    if is_none_allowed and uri is None:
        return
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


def _validate_transition_matrices(transition_matrices: Any, field_name: str) -> None:
    """Validate a transition matrices.

    Args:
        transition_matrices: A jax.Array with shape (n_states, n_states, n_actions).
        field_name: The name of the field.
    """
    if not isinstance(transition_matrices, jax.Array):
        raise ConfigValidationError(f"{field_name} must be a jax.Array, got {type(transition_matrices)}")
    if transition_matrices.ndim != 3:
        raise ConfigValidationError(f"{field_name} must be a 3D jax.Array, got {transition_matrices.shape}")
    if transition_matrices.shape[1] != transition_matrices.shape[2]:
        raise ConfigValidationError(
            f"{field_name} must have the same number of rows and columns, "
            f"got {transition_matrices.shape[1]} != {transition_matrices.shape[2]}"
        )
    if transition_matrices.dtype not in [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]:
        raise ConfigValidationError(f"{field_name} must be a float array, got {transition_matrices.dtype}")


def _validate_initial_state(initial_state: Any, n_states: int, field_name: str) -> None:
    """Validate an initial state.

    Args:
        initial_state: A jax.Array with shape (n_states,).
        n_states: The number of states in the transition matrices.
        field_name: The name of the field.
    """
    if not isinstance(initial_state, jax.Array):
        raise ConfigValidationError(f"{field_name} must be a jax.Array, got {type(initial_state)}")
    if initial_state.ndim != 1:
        raise ConfigValidationError(f"{field_name} must be a 1D jax.Array, got {initial_state.shape}")
    if initial_state.shape[0] != n_states:
        raise ConfigValidationError(
            f"{field_name} must have the same number of elements as the number of states in the transition matrices, "
            f"got {initial_state.shape[0]} != {n_states}"
        )
    if initial_state.dtype not in [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]:
        raise ConfigValidationError(f"{field_name} must be a float array, got {initial_state.dtype}")


# ============================================================================
# Instance Config
# ============================================================================


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str


def _validate_instance_config(cfg: DictConfig, expected_target: str | None = None) -> None:
    """Validate an InstanceConfig.

    Args:
        cfg: A DictConfig with an _target_ field (from Hydra).
        expected_target: The expected target, if any.
    """
    target = cfg.get("_target_", None)

    _validate_nonempty_str(target, "InstanceConfig._target_")
    if expected_target is not None and target != expected_target:
        raise ConfigValidationError(f"InstanceConfig._target_ must be {expected_target}, got {target}")


# ============================================================================
# MLflow Config
# ============================================================================


@dataclass
class MLFlowConfig:
    """Configuration for MLflow."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool | None = None


def validate_mlflow_config(cfg: DictConfig) -> None:
    """Validate an MLFlowConfig.

    Args:
        cfg: A DictConfig with MLFlowConfig fields (from Hydra).
    """
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")

    _validate_nonempty_str(experiment_id, "MLFlowConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowConfig.run_name", is_none_allowed=True)
    _validate_bool(downgrade_unity_catalog, "MLFlowConfig.downgrade_unity_catalog", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowConfig.registry_uri", is_none_allowed=True)


@dynamic_resolve
def update_mlflow_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a MLFlowConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")


# ============================================================================
# Logging Config
# ============================================================================


@dataclass
class FileLoggerInstanceConfig(InstanceConfig):
    """Configuration for FileLogger."""

    file_path: str

    def __init__(self, file_path: str, _target_: str = "simplexity.logging.file_logger.FileLogger") -> None:
        super().__init__(_target_=_target_)
        self.file_path = file_path


def is_file_logger_target(target: str) -> bool:
    """Check if the target is a file logger target."""
    return target == "simplexity.logging.file_logger.FileLogger"


def is_file_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a FileLoggerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_file_logger_target(target)
    return False


def validate_file_logger_instance_config(cfg: DictConfig) -> None:
    """Validate a FileLoggerInstanceConfig.

    Args:
        cfg: A DictConfig with FileLoggerInstanceConfig fields (from Hydra).
    """
    file_path = cfg.get("file_path")

    _validate_instance_config(cfg, expected_target="simplexity.logging.file_logger.FileLogger")
    _validate_nonempty_str(file_path, "FileLoggerInstanceConfig.file_path")


@dataclass
class MLFlowLoggerInstanceConfig(InstanceConfig):
    """Configuration for MLFlowLogger."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True


def is_mlflow_logger_target(target: str) -> bool:
    """Check if the target is a mlflow logger target."""
    return target == "simplexity.logging.mlflow_logger.MLFlowLogger"


def is_mlflow_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a MLFlowLoggerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_mlflow_logger_target(target)
    return False


def validate_mlflow_logger_instance_config(cfg: DictConfig) -> None:
    """Validate a MLFlowLoggerInstanceConfig.

    Args:
        cfg: A DictConfig with MLFlowLoggerInstanceConfig fields (from Hydra).
    """
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")

    _validate_instance_config(cfg, expected_target="simplexity.logging.mlflow_logger.MLFlowLogger")
    _validate_nonempty_str(experiment_id, "MLFlowLoggerInstanceConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowLoggerInstanceConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowLoggerInstanceConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowLoggerInstanceConfig.run_name", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowLoggerInstanceConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowLoggerInstanceConfig.registry_uri", is_none_allowed=True)
    _validate_bool(downgrade_unity_catalog, "MLFlowLoggerInstanceConfig.downgrade_unity_catalog", is_none_allowed=True)


@dynamic_resolve
def update_logging_instance_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a LoggingInstanceConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg._target_ = updated_cfg.get("_target_")  # pylint: disable=protected-access
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")


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
    name = cfg.get("name")

    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("LoggingConfig.instance must be a DictConfig")
    if not is_logger_config(instance):
        raise ConfigValidationError("LoggingConfig.instance must be a logger target")

    if is_file_logger_config(instance):
        validate_file_logger_instance_config(instance)
    elif is_mlflow_logger_config(instance):
        validate_mlflow_logger_instance_config(instance)
    else:
        _validate_instance_config(instance)
    _validate_nonempty_str(name, "LoggingConfig.name", is_none_allowed=True)


# ============================================================================
# Generative Process Config
# ============================================================================


@dataclass
class HiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the hidden markov model builder."""

    process_name: str
    process_params: Mapping[str, Any] | None = None
    initial_state: jax.Array | Sequence[float] | None = None

    def __init__(
        self,
        process_name: str,
        process_params: Mapping[str, Any] | None = None,
        initial_state: jax.Array | Sequence[float] | None = None,
        _target_: str = "simplexity.generative_processes.builder.build_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_name = process_name
        self.process_params = process_params
        self.initial_state = initial_state


def is_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_hidden_markov_model"


def is_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_hidden_markov_model_builder_target(target)
    return False


def validate_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a HiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with HiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_name = cfg.get("process_name")
    process_params = cfg.get("process_params")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(cfg, expected_target="simplexity.generative_processes.builder.build_hidden_markov_model")
    _validate_nonempty_str(process_name, "HiddenMarkovModelBuilderInstanceConfig.process_name")
    _validate_mapping(
        process_params, "HiddenMarkovModelBuilderInstanceConfig.process_params", key_type=str, is_none_allowed=True
    )
    _validate_sequence(
        initial_state, "HiddenMarkovModelBuilderInstanceConfig.initial_state", element_type=float, is_none_allowed=True
    )


@dataclass
class GeneralizedHiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the generalized hidden markov model builder."""

    process_name: str
    process_params: Mapping[str, Any] | None = None
    initial_state: jax.Array | Sequence[float] | None = None

    def __init__(
        self,
        process_name: str,
        process_params: Mapping[str, Any] | None = None,
        initial_state: jax.Array | Sequence[float] | None = None,
        _target_: str = "simplexity.generative_processes.builder.build_generalized_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_name = process_name
        self.process_params = process_params
        self.initial_state = initial_state


def is_generalized_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a generalized hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_generalized_hidden_markov_model"


def is_generalized_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generalized hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generalized_hidden_markov_model_builder_target(target)
    return False


def validate_generalized_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a GeneralizedHiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with GeneralizedHiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_name = cfg.get("process_name")
    process_params = cfg.get("process_params")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.builder.build_generalized_hidden_markov_model"
    )
    _validate_nonempty_str(process_name, "GeneralizedHiddenMarkovModelBuilderInstanceConfig.process_name")
    _validate_mapping(
        process_params,
        "GeneralizedHiddenMarkovModelBuilderInstanceConfig.process_params",
        key_type=str,
        is_none_allowed=True,
    )
    _validate_sequence(
        initial_state,
        "GeneralizedHiddenMarkovModelBuilderInstanceConfig.initial_state",
        element_type=float,
        is_none_allowed=True,
    )


@dataclass
class NonergodicHiddenMarkovModelBuilderInstanceConfig(InstanceConfig):
    """Configuration for the nonergodic hidden markov model builder."""

    process_names: list[str]
    process_params: Sequence[Mapping[str, Any]]
    process_weights: Sequence[float]
    vocab_maps: Sequence[Sequence[int]] | None = None
    add_bos_token: bool = False

    def __init__(
        self,
        process_names: list[str],
        process_params: Sequence[Mapping[str, Any]],
        process_weights: Sequence[float],
        vocab_maps: Sequence[Sequence[int]] | None = None,
        add_bos_token: bool = False,
        _target_: str = "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model",
    ) -> None:
        super().__init__(_target_=_target_)
        self.process_names = process_names
        self.process_params = process_params
        self.process_weights = process_weights
        self.vocab_maps = vocab_maps
        self.add_bos_token = add_bos_token


def is_nonergodic_hidden_markov_model_builder_target(target: str) -> bool:
    """Check if the target is a nonergodic hidden markov model builder target."""
    return target == "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model"


def is_nonergodic_hidden_markov_model_builder_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a nonergodic hidden markov model builder config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_nonergodic_hidden_markov_model_builder_target(target)
    return False


def validate_nonergodic_hidden_markov_model_builder_instance_config(cfg: DictConfig) -> None:
    """Validate a NonergodicHiddenMarkovModelBuilderInstanceConfig.

    Args:
        cfg: A DictConfig with NonergodicHiddenMarkovModelBuilderInstanceConfig fields (from Hydra).
    """
    process_names = cfg.get("process_names")
    process_params = cfg.get("process_params")
    process_weights = cfg.get("process_weights")
    vocab_maps = cfg.get("vocab_maps")
    add_bos_token = cfg.get("add_bos_token")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model"
    )
    if not isinstance(process_names, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names must be a list, got {type(process_names)}"
        )
    if not isinstance(process_params, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_params must be a sequence, "
            f"got {type(process_params)}"
        )
    if not isinstance(process_weights, Sequence):
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_weights must be a sequence, "
            f"got {type(process_weights)}"
        )
    if vocab_maps is None:
        _vocab_maps = [None] * len(process_names)
    else:
        if not isinstance(vocab_maps, Sequence):
            raise ConfigValidationError(
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.vocab_maps must be a sequence, "
                f"got {type(vocab_maps)}"
            )
        _vocab_maps = vocab_maps
    try:
        for i, (name, params, weights, vocab_map) in enumerate(
            zip(process_names, process_params, process_weights, _vocab_maps, strict=True)
        ):
            _validate_nonempty_str(name, f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names[{i}]")
            _validate_mapping(
                params,
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_params[{i}]",
                key_type=str,
                is_none_allowed=True,
            )
            _validate_non_negative_float(
                weights, f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_weights[{i}]"
            )
            _validate_sequence(
                vocab_map,
                f"NonergodicHiddenMarkovModelBuilderInstanceConfig.vocab_maps[{i}]",
                element_type=int,
                is_none_allowed=True,
            )
    except ValueError as e:
        var_str = "and process_weights" if vocab_maps is not None else "process_weights, and vocab_maps"
        vocab_len_str = f"!= {len(vocab_maps)}" if vocab_maps is not None else ""
        raise ConfigValidationError(
            f"NonergodicHiddenMarkovModelBuilderInstanceConfig.process_names, process_params, {var_str}"
            f"must have the same length, "
            f"got {len(process_names)} != {len(process_params)} != {len(process_weights)}{vocab_len_str}"
        ) from e
    _validate_bool(
        add_bos_token, "NonergodicHiddenMarkovModelBuilderInstanceConfig.add_bos_token", is_none_allowed=True
    )


@dataclass
class GeneralizedHiddenMarkovModelInstanceConfig(InstanceConfig):
    """Configuration for the generalized hidden markov model."""

    transition_matrices: jax.Array
    initial_state: jax.Array | None = None

    def __init__(
        self,
        transition_matrices: jax.Array,
        initial_state: jax.Array | None = None,
        _target_: str = "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel",
    ) -> None:
        super().__init__(_target_=_target_)
        self.transition_matrices = transition_matrices
        self.initial_state = initial_state


def is_generalized_hidden_markov_model_target(target: str) -> bool:
    """Check if the target is a generalized hidden markov model target."""
    return target == "simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel"


def is_generalized_hidden_markov_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a generalized hidden markov model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_generalized_hidden_markov_model_target(target)
    return False


def validate_generalized_hidden_markov_model_instance_config(cfg: DictConfig) -> None:
    """Validate a GeneralizedHiddenMarkovModelInstanceConfig.

    Args:
        cfg: A DictConfig with GeneralizedHiddenMarkovModelInstanceConfig fields (from Hydra).
    """
    transition_matrices = cfg.get("transition_matrices")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg,
        expected_target="simplexity.generative_processes.generalized_hidden_markov_model.GeneralizedHiddenMarkovModel",
    )
    _validate_transition_matrices(transition_matrices, "GeneralizedHiddenMarkovModelInstanceConfig.transition_matrices")
    assert isinstance(transition_matrices, jax.Array)
    _validate_initial_state(
        initial_state, transition_matrices.shape[1], "GeneralizedHiddenMarkovModelInstanceConfig.initial_state"
    )


@dataclass
class HiddenMarkovModelInstanceConfig(InstanceConfig):
    """Configuration for the hidden markov model."""

    transition_matrices: jax.Array
    initial_state: jax.Array | None = None

    def __init__(
        self,
        transition_matrices: jax.Array,
        initial_state: jax.Array | None = None,
        _target_: str = "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel",
    ) -> None:
        super().__init__(_target_=_target_)
        self.transition_matrices = transition_matrices
        self.initial_state = initial_state


def is_hidden_markov_model_target(target: str) -> bool:
    """Check if the target is a hidden markov model target."""
    return target.startswith("simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel")


def is_hidden_markov_model_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a hidden markov model config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_hidden_markov_model_target(target)
    return False


def validate_hidden_markov_model_instance_config(cfg: DictConfig) -> None:
    """Validate a HiddenMarkovModelInstanceConfig.

    Args:
        cfg: A DictConfig with HiddenMarkovModelInstanceConfig fields (from Hydra).
    """
    transition_matrices = cfg.get("transition_matrices")
    initial_state = cfg.get("initial_state")

    _validate_instance_config(
        cfg, expected_target="simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel"
    )
    _validate_transition_matrices(transition_matrices, "HiddenMarkovModelInstanceConfig.transition_matrices")
    assert isinstance(transition_matrices, jax.Array)
    _validate_initial_state(
        initial_state, transition_matrices.shape[1], "HiddenMarkovModelInstanceConfig.initial_state"
    )


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
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("GenerativeProcessConfig.instance is required")

    if is_generalized_hidden_markov_model_builder_config(instance):
        validate_generalized_hidden_markov_model_builder_instance_config(instance)
    elif is_hidden_markov_model_builder_config(instance):
        validate_hidden_markov_model_builder_instance_config(instance)
    else:
        _validate_instance_config(instance)
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
class LocalPersisterInstanceConfig(InstanceConfig):
    """Configuration for the local persister."""

    directory: str

    def __init__(self, directory: str, _target_: str = "simplexity.persistence.local_persister.LocalPersister"):
        super().__init__(_target_=_target_)
        self.directory = directory


def is_local_persister_config(cfg: DictConfig, framework: str | None = None) -> bool:
    """Check if the configuration is a LocalPersisterInstanceConfig."""
    if framework is None:
        file_pattern = "local_[a-z]+_persister"
        class_pattern = "Local[A-Z][a-z]+Persister"
    else:
        file_pattern = f"local_{framework.lower()}_persister"
        class_pattern = f"Local{framework.capitalize()}Persister"
    return (
        re.match(f"simplexity.persistence.{file_pattern}.{class_pattern}", OmegaConf.select(cfg, "_target_"))
        is not None
    )


def validate_local_persister_instance_config(cfg: DictConfig, framework: str | None = None) -> None:
    """Validate a LocalPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPersisterInstanceConfig fields (from Hydra).
        framework: The framework of the local persister. If None, the framework will be inferred from the target.
    """
    target = cfg.get("_target_")
    directory = cfg.get("directory")

    _validate_instance_config(cfg)
    if not is_local_persister_config(cfg, framework=framework):
        class_name = f"Local{framework.capitalize()}Persister" if framework is not None else "LocalPersister"
        raise ConfigValidationError(f"{class_name}InstanceConfig must be a local persister, got {target}")
    _validate_nonempty_str(directory, "LocalPersisterInstanceConfig.directory")


@dataclass
class LocalEquinoxPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local equinox persister."""

    filename: str = "model.eqx"

    def __init__(
        self,
        directory: str,
        filename: str = "model.eqx",
        _target_: str = "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
    ):
        super().__init__(_target_=_target_, directory=directory)
        self.filename = filename


def is_local_equinox_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalEquinoxPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="equinox")


def validate_local_equinox_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalEquinoxPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalEquinoxPersisterInstanceConfig fields (from Hydra).
    """
    filename = cfg.get("filename")

    validate_local_persister_instance_config(cfg, framework="equinox")
    _validate_nonempty_str(filename, "LocalEquinoxPersisterInstanceConfig.filename")
    if not filename.endswith(".eqx"):
        raise ConfigValidationError("LocalEquinoxPersisterInstanceConfig.filename must end with .eqx, got {filename}")


@dataclass
class LocalPenzaiPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local penzai persister."""

    def __init__(
        self, directory: str, _target_: str = "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister"
    ):
        super().__init__(_target_=_target_, directory=directory)


def is_local_penzai_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalPenzaiPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="penzai")


def validate_local_penzai_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalPenzaiPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPenzaiPersisterInstanceConfig fields (from Hydra).
    """
    validate_local_persister_instance_config(cfg, framework="penzai")


@dataclass
class LocalPytorchPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local pytorch persister."""

    filename: str = "model.pt"

    def __init__(
        self,
        directory: str,
        filename: str = "model.pt",
        _target_: str = "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister",
    ):
        super().__init__(_target_=_target_, directory=directory)
        self.filename = filename


def is_local_pytorch_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalPytorchPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="pytorch")


def validate_local_pytorch_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalPytorchPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPytorchPersisterInstanceConfig fields (from Hydra).
    """
    filename = cfg.get("filename")

    validate_local_persister_instance_config(cfg, framework="pytorch")
    _validate_nonempty_str(filename, "LocalPytorchPersisterInstanceConfig.filename")
    if not filename.endswith(".pt"):
        raise ConfigValidationError("LocalPytorchPersisterInstanceConfig.filename must end with .pt, got {filename}")


@dataclass
class MLFlowPersisterInstanceConfig(InstanceConfig):
    """Configuration for the MLflow persister."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True
    artifact_path: str | None = "models"
    config_path: str | None = "config.yaml"

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        artifact_path: str | None = "models",
        config_path: str | None = "config.yaml",
        _target_: str = "simplexity.persistence.mlflow_persister.MLFlowPersister",
    ):
        super().__init__(_target_=_target_)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.downgrade_unity_catalog = downgrade_unity_catalog
        self.artifact_path = artifact_path
        self.config_path = config_path


def is_mlflow_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a MLFlowPersisterInstanceConfig."""
    return OmegaConf.select(cfg, "_target_") == "simplexity.persistence.mlflow_persister.MLFlowPersister"


def validate_mlflow_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a MLFlowPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with MLFlowPersisterInstanceConfig fields (from Hydra).
    """
    _validate_instance_config(cfg, expected_target="simplexity.persistence.mlflow_persister.MLFlowPersister")
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")
    artifact_path = cfg.get("artifact_path")
    config_path = cfg.get("config_path")

    _validate_nonempty_str(experiment_id, "MLFlowPersisterInstanceConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowPersisterInstanceConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowPersisterInstanceConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowPersisterInstanceConfig.run_name", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowPersisterInstanceConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowPersisterInstanceConfig.registry_uri", is_none_allowed=True)
    _validate_bool(
        downgrade_unity_catalog, "MLFlowPersisterInstanceConfig.downgrade_unity_catalog", is_none_allowed=True
    )
    _validate_nonempty_str(artifact_path, "MLFlowPersisterInstanceConfig.artifact_path", is_none_allowed=True)
    _validate_nonempty_str(config_path, "MLFlowPersisterInstanceConfig.config_path", is_none_allowed=True)


@dynamic_resolve
def update_persister_instance_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a PersistenceConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg._target_ = updated_cfg.get("_target_")  # pylint: disable=protected-access
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")
    cfg.artifact_path = updated_cfg.get("artifact_path")
    cfg.config_path = updated_cfg.get("config_path")


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
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("PersistenceConfig.instance is required")
    if is_local_equinox_persister_config(instance):
        validate_local_equinox_persister_instance_config(instance)
    elif is_local_penzai_persister_config(instance):
        validate_local_penzai_persister_instance_config(instance)
    elif is_local_pytorch_persister_config(instance):
        validate_local_pytorch_persister_instance_config(instance)
    elif is_mlflow_persister_config(instance):
        validate_mlflow_persister_instance_config(instance)
    else:
        _validate_instance_config(instance)
    _validate_nonempty_str(cfg.get("name"), "PersistenceConfig.name", is_none_allowed=True)


# ============================================================================
# Predictive Model Configs
# ============================================================================


@dataclass
class HookedTransformerConfigConfig(InstanceConfig):
    """Configuration for HookedTransformerConfig."""

    n_layers: int
    d_model: int
    d_head: int
    n_ctx: int
    n_heads: int = -1
    d_mlp: int | None = None
    act_fn: str | None = None
    d_vocab: int = MISSING
    normalization_type: str | None = "LN"
    device: str | None = None
    seed: int | None = None

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        n_ctx: int,
        n_heads: int = -1,
        d_mlp: int | None = None,
        act_fn: str | None = None,
        d_vocab: int = MISSING,
        normalization_type: str | None = "LN",
        device: str | None = None,
        seed: int | None = None,
        _target_: str = "transformer_lens.HookedTransformerConfig",
    ):
        super().__init__(_target_=_target_)
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.act_fn = act_fn
        self.d_vocab = d_vocab
        self.normalization_type = normalization_type
        self.device = device
        self.seed = seed


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
    act_fn = cfg.get("act_fn")
    normalization_type = cfg.get("normalization_type")
    device = cfg.get("device")
    seed = cfg.get("seed")

    _validate_instance_config(cfg, expected_target="transformer_lens.HookedTransformerConfig")
    _validate_positive_int(n_layers, "HookedTransformerConfigConfig.n_layers")
    _validate_positive_int(d_model, "HookedTransformerConfigConfig.d_model")
    _validate_positive_int(d_head, "HookedTransformerConfigConfig.d_head")
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
    _validate_nonempty_str(act_fn, "HookedTransformerConfigConfig.act_fn", is_none_allowed=True)
    if OmegaConf.is_missing(cfg, "d_vocab"):
        SIMPLEXITY_LOGGER.debug("[predictive model] d_vocab is missing, will be resolved dynamically")
    else:
        d_vocab = cfg.get("d_vocab")
        _validate_positive_int(d_vocab, "HookedTransformerConfigConfig.d_vocab")
    _validate_nonempty_str(normalization_type, "HookedTransformerConfigConfig.normalization_type", is_none_allowed=True)
    _validate_nonempty_str(device, "HookedTransformerConfigConfig.device", is_none_allowed=True)
    _validate_non_negative_int(seed, "HookedTransformerConfigConfig.seed", is_none_allowed=True)


@dataclass
class HookedTransformerInstancecConfig(InstanceConfig):
    """Configuration for Transformer model."""

    cfg: HookedTransformerConfigConfig

    def __init__(self, cfg: HookedTransformerConfigConfig, _target_: str = "transformer_lens.HookedTransformer"):
        super().__init__(_target_=_target_)
        self.cfg = cfg


def is_hooked_transformer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a HookedTransformerConfig."""
    return OmegaConf.select(cfg, "_target_") == "transformer_lens.HookedTransformer"


def validate_hooked_transformer_config(cfg: DictConfig) -> None:
    """Validate a HookedTransformerInstancecConfig.

    Args:
        cfg: A DictConfig with _target_ and cfg fields (from Hydra).
    """
    _validate_instance_config(cfg)
    nested_cfg = cfg.get("cfg")
    if nested_cfg is None:
        raise ConfigValidationError("HookedTransformerConfig.cfg is required")
    validate_hooked_transformer_config_config(nested_cfg)


@dynamic_resolve
def resolve_hooked_transformer_config(cfg: DictConfig, *, vocab_size: int | None = None) -> None:
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


def validate_predictive_model_config(cfg: DictConfig) -> None:
    """Validate the configuration.

    Args:
        cfg: A DictConfig with instance, optional name, and optional load_checkpoint_step fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("PredictiveModelConfig.instance is required")
    target = instance.get("_target_", None)
    name = cfg.get("name")
    load_checkpoint_step = cfg.get("load_checkpoint_step")

    _validate_instance_config(instance)
    if not is_predictive_model_target(target):
        raise ConfigValidationError(
            f"PredictiveModelConfig.instance._target_ must be a predictive model target, got {target}"
        )
    if is_hooked_transformer_config(cfg):
        validate_hooked_transformer_config(instance)
    _validate_nonempty_str(name, "PredictiveModelConfig.name", is_none_allowed=True)
    _validate_non_negative_int(load_checkpoint_step, "PredictiveModelConfig.load_checkpoint_step", is_none_allowed=True)


# ============================================================================
# Optimizer Config
# ============================================================================


@dataclass
class AdamInstanceConfig(InstanceConfig):
    """Configuration for the Adam optimizer."""

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


def is_pytorch_adam_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.adam")
    return False


def validate_adam_instance_config(cfg: DictConfig) -> None:
    """Validate an AdamInstanceConfig.

    Args:
        cfg: A DictConfig with AdamInstanceConfig fields (from Hydra).
    """
    _validate_instance_config(cfg)
    lr = cfg.get("lr")
    betas = cfg.get("betas")
    eps = cfg.get("eps")
    weight_decay = cfg.get("weight_decay")
    amsgrad = cfg.get("amsgrad")

    _validate_positive_float(lr, "AdamInstanceConfig.lr", is_none_allowed=True)
    if betas is not None:
        if len(betas) != 2:
            raise ConfigValidationError(f"AdamInstanceConfig.betas must have length 2, got {len(betas)}")
        _validate_non_negative_float(betas[0], "AdamInstanceConfig.betas[0]")
        _validate_non_negative_float(betas[1], "AdamInstanceConfig.betas[1]")
    _validate_non_negative_float(eps, "AdamInstanceConfig.eps", is_none_allowed=True)
    _validate_non_negative_float(weight_decay, "AdamInstanceConfig.weight_decay", is_none_allowed=True)
    _validate_bool(amsgrad, "AdamInstanceConfig.amsgrad", is_none_allowed=True)


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
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("OptimizerConfig.instance must be a DictConfig")
    target = instance.get("_target_")
    name = cfg.get("name")

    _validate_instance_config(instance)
    if not is_optimizer_target(target):
        raise ConfigValidationError(f"OptimizerConfig.instance._target_ must be an optimizer target, got {target}")
    elif is_pytorch_adam_optimizer_config(instance):
        validate_adam_instance_config(instance)
    _validate_nonempty_str(name, "OptimizerConfig.name", is_none_allowed=True)


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
    tags = cfg.get("tags")
    mlflow = cfg.get("mlflow")

    _validate_non_negative_int(seed, "BaseConfig.seed", is_none_allowed=True)
    _validate_mapping(tags, "BaseConfig.tags", key_type=str, value_type=str, is_none_allowed=True)
    if mlflow is not None:
        if not isinstance(mlflow, DictConfig):
            raise ConfigValidationError("BaseConfig.mlflow must be a MLFlowConfig")
        validate_mlflow_config(mlflow)
