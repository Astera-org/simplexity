"""Field resolution from projections, scalars, and belief states."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from simplexity.activations.visualization_configs import ActivationVisualizationFieldRef
from simplexity.exceptions import ConfigValidationError


def _lookup_projection_array(
    projections: Mapping[str, np.ndarray], layer_name: str, key: str | None, concat_layers: bool
) -> np.ndarray:
    """Look up a projection array by key, handling layer naming conventions."""
    if key is None:
        raise ConfigValidationError("Projection references must supply a `key` value.")
    suffix = f"_{key}"
    for full_key, value in projections.items():
        if concat_layers:
            if full_key.endswith(suffix) or full_key == key:
                return np.asarray(value)
        else:
            if not full_key.endswith(suffix):
                continue
            candidate_layer = full_key[: -len(suffix)]
            if candidate_layer == layer_name:
                return np.asarray(value)
    raise ConfigValidationError(f"Projection '{key}' not available for layer '{layer_name}'.")


def _lookup_scalar_value(scalars: Mapping[str, float], layer_name: str, key: str, concat_layers: bool) -> float:
    """Look up a scalar value by key, handling layer naming conventions."""
    suffix = f"_{key}"
    for full_key, value in scalars.items():
        if concat_layers:
            if full_key.endswith(suffix) or full_key == key:
                return float(value)
        else:
            if full_key.endswith(suffix) and full_key[: -len(suffix)] == layer_name:
                return float(value)
    raise ConfigValidationError(f"Scalar '{key}' not available for layer '{layer_name}'.")


def _maybe_component(array: np.ndarray, component: int | None) -> np.ndarray:
    """Extract a component from a 2D array, or return the 1D array as-is."""
    np_array = np.asarray(array)
    if np_array.ndim == 1:
        if component is not None:
            raise ConfigValidationError("Component index is invalid for 1D projection arrays.")
        return np_array
    if np_array.ndim != 2:
        raise ConfigValidationError("Projection arrays must be 1D or 2D.")
    if component is None:
        raise ConfigValidationError("Projection references for 2D arrays must specify `component`.")
    if component < 0 or component >= np_array.shape[1]:
        raise ConfigValidationError(
            f"Component index {component} is out of bounds for projection dimension {np_array.shape[1]}"
        )
    return np_array[:, component]


def _resolve_belief_states(belief_states: np.ndarray, ref: ActivationVisualizationFieldRef) -> np.ndarray:
    """Resolve belief states to a 1D array based on field reference configuration."""
    np_array = np.asarray(belief_states)

    # Handle factor dimension for 3D belief states (samples, factors, states)
    if np_array.ndim == 3:
        if ref.factor is None:
            raise ConfigValidationError(
                f"Belief states have 3 dimensions (samples, factors, states) but no `factor` was specified. "
                f"Shape: {np_array.shape}"
            )
        assert not isinstance(ref.factor, str), "Factor patterns should be expanded before resolution"
        factor_idx = ref.factor
        if factor_idx < 0 or factor_idx >= np_array.shape[1]:
            raise ConfigValidationError(
                f"Belief state factor {factor_idx} is out of bounds for dimension {np_array.shape[1]}"
            )
        np_array = np_array[:, factor_idx, :]  # Now 2D: (samples, states)
    elif np_array.ndim == 2:
        if ref.factor is not None:
            raise ConfigValidationError(
                f"Belief states are 2D but `factor={ref.factor}` was specified. "
                f"Factor selection requires 3D belief states (samples, factors, states)."
            )
    else:
        raise ConfigValidationError(f"Belief states must be 2D or 3D, got {np_array.ndim}D")

    # Now np_array is 2D: (samples, states)
    if ref.reducer == "argmax":
        return np.argmax(np_array, axis=1)
    if ref.reducer == "l2_norm":
        return np.linalg.norm(np_array, axis=1)
    assert not isinstance(ref.component, str), "Component patterns should be expanded before resolution"
    component = ref.component if ref.component is not None else 0
    if component < 0 or component >= np_array.shape[1]:
        raise ConfigValidationError(
            f"Belief state component {component} is out of bounds for dimension {np_array.shape[1]}"
        )
    return np_array[:, component]


def _resolve_field(
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    num_rows: int,
    metadata_columns: Mapping[str, object],
) -> np.ndarray:
    """Resolve a field reference to a numpy array of values."""
    if ref.source == "metadata":
        if ref.key is None:
            raise ConfigValidationError("Metadata references must specify `key`.")
        if ref.key == "layer":
            return np.repeat(layer_name, num_rows)
        if ref.key not in metadata_columns:
            raise ConfigValidationError(f"Metadata column '{ref.key}' is not available.")
        return np.asarray(metadata_columns[ref.key])

    if ref.source == "weights":
        if "weight" not in metadata_columns:
            raise ConfigValidationError("Weight metadata is unavailable for visualization mapping.")
        return np.asarray(metadata_columns["weight"])

    if ref.source == "projections":
        array = _lookup_projection_array(projections, layer_name, ref.key, analysis_concat_layers)
        assert not isinstance(ref.component, str), "Component patterns should be expanded before resolution"
        return _maybe_component(array, ref.component)

    if ref.source == "belief_states":
        if belief_states is None:
            raise ConfigValidationError("Visualization requests belief_states but they were not retained.")
        return _resolve_belief_states(belief_states, ref)

    if ref.source == "scalars":
        if ref.key is None:
            raise ConfigValidationError("Scalar references must supply `key`.")
        value = _lookup_scalar_value(scalars, layer_name, ref.key, analysis_concat_layers)
        return np.repeat(value, num_rows)

    raise ConfigValidationError(f"Unsupported field source '{ref.source}'")


__all__ = [
    "_lookup_projection_array",
    "_lookup_scalar_value",
    "_maybe_component",
    "_resolve_belief_states",
    "_resolve_field",
]
