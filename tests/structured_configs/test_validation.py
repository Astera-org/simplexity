"""Unit tests for structured config validation helpers."""

from __future__ import annotations

import types
from collections import OrderedDict

import jax.numpy as jnp
import pytest

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs import validation as validation_mod
from simplexity.structured_configs.validation import (
    validate_bool,
    validate_initial_state,
    validate_mapping,
    validate_non_negative_float,
    validate_non_negative_int,
    validate_nonempty_str,
    validate_positive_float,
    validate_positive_int,
    validate_sequence,
    validate_transition_matrices,
    validate_uri,
)


def test_validate_nonempty_str_success() -> None:
    validate_nonempty_str("ok", "field")
    validate_nonempty_str("something", "field", is_none_allowed=True)
    validate_nonempty_str(None, "field", is_none_allowed=True)


def test_validate_nonempty_str_errors() -> None:
    with pytest.raises(ConfigValidationError, match="field must be a string"):
        validate_nonempty_str(123, "field")
    with pytest.raises(ConfigValidationError, match="field must be a non-empty string"):
        validate_nonempty_str("   ", "field")


@pytest.mark.parametrize(
    "validator,value,error",
    [
        (validate_positive_int, "1", "must be an int"),
        (validate_positive_int, 0, "must be positive"),
        (validate_non_negative_int, "0", "must be an int"),
        (validate_non_negative_int, False, "must be an int"),
        (validate_non_negative_int, -1, "must be non-negative"),
    ],
)
def test_integer_validators_raise(validator, value, error) -> None:
    with pytest.raises(ConfigValidationError, match=error):
        validator(value, "field")


def test_integer_validators_allow_valid_values() -> None:
    validate_positive_int(3, "field")
    validate_non_negative_int(0, "field")
    validate_non_negative_int(None, "field", is_none_allowed=True)


@pytest.mark.parametrize(
    "validator,value,error",
    [
        (validate_positive_float, "0.1", "must be a float"),
        (validate_positive_float, -0.5, "must be positive"),
        (validate_non_negative_float, "0.1", "must be a float"),
        (validate_non_negative_float, -0.1, "must be non-negative"),
    ],
)
def test_float_validators_raise(validator, value, error) -> None:
    with pytest.raises(ConfigValidationError, match=error):
        validator(value, "field")


def test_float_validators_allow_valid_values() -> None:
    validate_positive_float(0.1, "field")
    validate_non_negative_float(0.0, "field")
    validate_non_negative_float(None, "field", is_none_allowed=True)


def test_validate_bool() -> None:
    validate_bool(True, "flag")
    validate_bool(False, "flag")
    validate_bool(None, "flag", is_none_allowed=True)
    with pytest.raises(ConfigValidationError, match="flag must be a bool"):
        validate_bool("true", "flag")


@pytest.mark.parametrize(
    "value,kwargs",
    [([0.1, 0.2], {"element_type": float}), ([1, 2], {}), (jnp.ones((2,), dtype=jnp.float32), {"element_type": float})],
)
def test_validate_sequence_accepts_valid_inputs(value, kwargs) -> None:
    validate_sequence(value, "seq", **kwargs)


def test_validate_sequence_rejects_bad_inputs() -> None:
    with pytest.raises(ConfigValidationError, match="seq must be a sequence"):
        validate_sequence(123, "seq")
    with pytest.raises(ConfigValidationError, match="seq must be a 1D array"):
        validate_sequence(jnp.ones((2, 2), dtype=jnp.float32), "seq", element_type=float)
    with pytest.raises(ConfigValidationError, match="seq must be a float array"):
        validate_sequence(jnp.ones((2,), dtype=jnp.int32), "seq", element_type=float)
    with pytest.raises(ConfigValidationError, match="seq items must be floats"):
        validate_sequence([1, "bad"], "seq", element_type=int)


@pytest.mark.parametrize(
    "value,kwargs",
    [
        (OrderedDict({"k": "v"}), {"key_type": str, "value_type": str}),
        ({"k": 1}, {"key_type": str}),
        (None, {"is_none_allowed": True}),
    ],
)
def test_validate_mapping_accepts_valid_inputs(value, kwargs) -> None:
    validate_mapping(value, "mapping", **kwargs)


def test_validate_mapping_rejects_bad_inputs() -> None:
    with pytest.raises(ConfigValidationError, match="mapping must be a dictionary"):
        validate_mapping(123, "mapping")
    with pytest.raises(ConfigValidationError, match="mapping keys must be strs"):
        validate_mapping({123: "v"}, "mapping", key_type=str)
    with pytest.raises(ConfigValidationError, match="mapping values must be ints"):
        validate_mapping({"k": "v"}, "mapping", value_type=int)


def test_validate_uri_allows_expected_schemes(monkeypatch) -> None:
    validate_uri("databricks://workspace", "uri")
    validate_uri("file:///tmp/file", "uri")
    validate_uri(None, "uri", is_none_allowed=True)

    # Force urlparse to raise so the error branch is covered.
    def boom(_uri: str):  # pragma: no cover - exercised via test
        raise ValueError("boom")

    monkeypatch.setattr(validation_mod, "urlparse", boom)
    with pytest.raises(ConfigValidationError, match="uri is not a valid URI: boom"):
        validate_uri("http://example.com", "uri")


def test_validate_uri_rejects_invalid_strings(monkeypatch) -> None:
    with pytest.raises(ConfigValidationError, match="uri cannot be empty"):
        validate_uri("   ", "uri")
    with pytest.raises(ConfigValidationError, match="uri must have a valid URI scheme"):
        validate_uri("relative/path", "uri")


def test_validate_transition_matrices_success() -> None:
    matrices = jnp.ones((2, 2, 2), dtype=jnp.float32)
    validate_transition_matrices(matrices, "matrices")


def test_validate_transition_matrices_failures() -> None:
    with pytest.raises(ConfigValidationError, match="matrices must be a jax.Array"):
        validate_transition_matrices([[1]], "matrices")
    with pytest.raises(ConfigValidationError, match="matrices must be a 3D jax.Array"):
        validate_transition_matrices(jnp.ones((2, 2), dtype=jnp.float32), "matrices")
    with pytest.raises(ConfigValidationError, match="must have the same number of rows and columns"):
        validate_transition_matrices(jnp.ones((2, 3, 1), dtype=jnp.float32), "matrices")
    with pytest.raises(ConfigValidationError, match="must be a float array"):
        validate_transition_matrices(jnp.ones((2, 2, 2), dtype=jnp.int32), "matrices")


def test_validate_initial_state_success() -> None:
    state = jnp.ones((2,), dtype=jnp.float32)
    validate_initial_state(state, 2, "state")


def test_validate_initial_state_failures() -> None:
    with pytest.raises(ConfigValidationError, match="state must be a jax.Array"):
        validate_initial_state([1, 2], 2, "state")
    with pytest.raises(ConfigValidationError, match="state must be a 1D jax.Array"):
        validate_initial_state(jnp.ones((2, 1), dtype=jnp.float32), 2, "state")
    with pytest.raises(ConfigValidationError, match="state must have the same number of elements"):
        validate_initial_state(jnp.ones((3,), dtype=jnp.float32), 2, "state")
    with pytest.raises(ConfigValidationError, match="state must be a float array"):
        validate_initial_state(jnp.ones((2,), dtype=jnp.int32), 2, "state")
