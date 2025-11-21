"""Validation helper functions for structured configs."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import jax
import jax.numpy as jnp

from simplexity.exceptions import ConfigValidationError


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

