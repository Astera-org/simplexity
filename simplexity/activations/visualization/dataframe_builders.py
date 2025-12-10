"""DataFrame construction for activation visualizations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from simplexity.activations.visualization.data_structures import (
    _SCALAR_INDEX_SENTINEL,
    PreparedMetadata,
)
from simplexity.activations.visualization.field_resolution import _resolve_field
from simplexity.activations.visualization.pattern_expansion import (
    _expand_field_mapping,
    _expand_scalar_pattern_keys,
    _scalar_pattern_label,
)
from simplexity.activations.visualization.pattern_utils import has_pattern
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationFieldRef,
    ScalarSeriesMapping,
)
from simplexity.exceptions import ConfigValidationError


def _build_metadata_columns(
    analysis_name: str,
    metadata: PreparedMetadata,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Build base metadata columns for visualization DataFrames."""
    sequences = metadata.sequences
    numeric_steps = metadata.steps
    sequence_strings = [" ".join(str(token) for token in seq) for seq in sequences]
    base = {
        "analysis": np.repeat(analysis_name, len(sequences)),
        "step": numeric_steps,
        "sequence_length": numeric_steps,
        "sequence": np.asarray(sequence_strings),
        "sample_index": np.arange(len(sequences), dtype=np.int32),
        "weight": weights,
    }
    return base


def _extract_base_column_name(column: str, group_value: str, original_field_pattern: str | None) -> str:
    """Extract base column name by removing group index from expanded column name.

    For column='factor_0_prob_0' with group_value='0', returns 'prob_0'.
    Uses the group_value to identify and remove the group-related part.

    In practice, key-expanded columns will have format prefix_N_suffix where prefix
    is the group name (e.g., 'factor') and suffix is the base column (e.g., 'prob_0').
    Columns like 'prob_0' without a clear group prefix are returned unchanged.
    """
    # Pattern: prefix_N_suffix (e.g., factor_0_prob_0 -> prob_0)
    # Must have alphabetic suffix after the group value underscore to ensure
    # we're stripping a real group prefix, not just matching any column ending in _N
    pattern = re.compile(rf"^([a-zA-Z][a-zA-Z0-9_]*)_{re.escape(group_value)}_([a-zA-Z].*)$")
    match = pattern.match(column)
    if match:
        return match.group(2)

    # No match - return original column unchanged
    # This handles cases like 'prob_0' where there's no group prefix to strip
    return column


def _build_scalar_dataframe(
    mappings: dict[str, ActivationVisualizationFieldRef],
    scalars: Mapping[str, float],
    scalar_history: Mapping[str, list[tuple[int, float]]],
    analysis_name: str,
    current_step: int,
) -> pd.DataFrame:
    """Build a long-format DataFrame for scalar visualizations supporting both current and historical data."""
    rows: list[dict[str, Any]] = []

    for field_name, ref in mappings.items():
        if ref.source not in ("scalar_pattern", "scalar_history"):
            continue

        if ref.key is None:
            raise ConfigValidationError(f"{ref.source} field references must specify a key")

        # Determine which scalar keys this mapping should include
        if has_pattern(ref.key):
            # Match pattern against both current scalars and history keys
            all_available_keys = set(scalars.keys()) | set(scalar_history.keys())
            matched_keys = _expand_scalar_pattern_keys(ref.key, all_available_keys, analysis_name)
        else:
            matched_keys = [ref.key if "/" in ref.key else f"{analysis_name}/{ref.key}"]

        for scalar_key in matched_keys:
            if ref.source == "scalar_pattern":
                # scalar_pattern: Always use current scalar values
                # This ensures compatibility with accumulate_steps file persistence
                if scalar_key in scalars:
                    value = scalars[scalar_key]
                    rows.append(
                        {
                            "step": current_step,
                            "layer": _scalar_pattern_label(scalar_key),
                            field_name: value,
                            "metric": scalar_key,
                        }
                    )
            elif ref.source == "scalar_history":
                # scalar_history: Use full in-memory history
                if scalar_key in scalar_history and scalar_history[scalar_key]:
                    for step, value in scalar_history[scalar_key]:
                        rows.append(
                            {
                                "step": step,
                                "layer": _scalar_pattern_label(scalar_key),
                                field_name: value,
                                "metric": scalar_key,
                            }
                        )
                elif scalar_key in scalars:
                    # No history yet, use current value
                    value = scalars[scalar_key]
                    rows.append(
                        {
                            "step": current_step,
                            "layer": _scalar_pattern_label(scalar_key),
                            field_name: value,
                            "metric": scalar_key,
                        }
                    )

    if not rows:
        raise ConfigValidationError(
            "Scalar visualization could not find any matching scalar values. "
            f"Available keys: {list(scalars.keys())}, History keys: {list(scalar_history.keys())}"
        )

    return pd.DataFrame(rows)


def _build_scalar_series_dataframe(
    mapping: ScalarSeriesMapping,
    metadata_columns: Mapping[str, Any],
    scalars: Mapping[str, float],
    layer_names: list[str],
    analysis_name: str,
) -> pd.DataFrame:
    """Build a DataFrame from scalar series data."""
    base_metadata = _scalar_series_metadata(metadata_columns)
    rows: list[dict[str, Any]] = []
    for layer_name in layer_names:
        index_values = mapping.index_values or _infer_scalar_series_indices(mapping, scalars, layer_name, analysis_name)
        for index_value in index_values:
            raw_key = mapping.key_template.format(layer=layer_name, index=index_value)
            scalar_key = f"{analysis_name}/{raw_key}"
            scalar_value = scalars.get(scalar_key)
            if scalar_value is None:
                continue
            row: dict[str, Any] = {
                mapping.index_field: index_value,
                mapping.value_field: scalar_value,
                "layer": layer_name,
            }
            row.update(base_metadata)
            rows.append(row)
    if not rows:
        raise ConfigValidationError(
            "Scalar series visualization could not resolve any scalar values with the provided key_template."
        )
    return pd.DataFrame(rows)


def _infer_scalar_series_indices(
    mapping: ScalarSeriesMapping,
    scalars: Mapping[str, float],
    layer_name: str,
    analysis_name: str,
) -> list[int]:
    """Infer available indices for scalar series from available scalar keys."""
    raw_template = mapping.key_template.format(layer=layer_name, index=_SCALAR_INDEX_SENTINEL)
    template = f"{analysis_name}/{raw_template}"
    if _SCALAR_INDEX_SENTINEL not in template:
        raise ConfigValidationError(
            "scalar_series.key_template must include '{index}' placeholder to infer index values."
        )
    prefix, suffix = template.split(_SCALAR_INDEX_SENTINEL, 1)
    inferred: set[int] = set()
    for key in scalars:
        if not key.startswith(prefix):
            continue
        if suffix and not key.endswith(suffix):
            continue
        body = key[len(prefix) : len(key) - len(suffix) if suffix else None]
        if not body:
            continue
        try:
            inferred.add(int(body))
        except ValueError:
            continue
    if not inferred:
        raise ConfigValidationError(
            f"Scalar series could not infer indices for layer '{layer_name}' "
            f"using key_template '{mapping.key_template}'."
        )
    return sorted(inferred)


def _scalar_series_metadata(metadata_columns: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scalar metadata from metadata columns."""
    metadata: dict[str, Any] = {}
    for key, value in metadata_columns.items():
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            metadata[key] = value.flat[0]
        else:
            metadata[key] = value
    return metadata


def _build_dataframe_for_mappings(
    mappings: dict[str, ActivationVisualizationFieldRef],
    metadata_columns: Mapping[str, Any],
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    layer_names: list[str],
) -> pd.DataFrame:
    """Build a DataFrame from a single set of mappings (used by both regular and combined modes)."""
    base_rows = len(metadata_columns["step"])
    frames: list[pd.DataFrame] = []

    # Check if mappings are belief-state-only (don't need layer iteration)
    all_belief_states = all(ref.source == "belief_states" for ref in mappings.values())
    effective_layer_names = ["_no_layer_"] if all_belief_states else layer_names

    for layer_name in effective_layer_names:
        # Expand all mappings first
        expanded_mappings: dict[str, ActivationVisualizationFieldRef] = {}
        for field_name, ref in mappings.items():
            try:
                expanded = _expand_field_mapping(
                    field_name, ref, layer_name, projections, scalars, belief_states, analysis_concat_layers
                )
                expanded_mappings.update(expanded)
            except ConfigValidationError as e:
                raise ConfigValidationError(f"Error expanding '{field_name}' for layer '{layer_name}': {e}") from e

        # Check if any refs have group expansion (_group_value set)
        group_refs = {col: ref for col, ref in expanded_mappings.items() if ref._group_value is not None}
        non_group_refs = {col: ref for col, ref in expanded_mappings.items() if ref._group_value is None}

        if group_refs:
            # Group expansion: restructure to long format
            # Group refs by _group_value
            groups: dict[str, dict[str, ActivationVisualizationFieldRef]] = {}
            group_column_name: str | None = None

            for col, ref in group_refs.items():
                group_val = ref._group_value
                assert group_val is not None
                if group_val not in groups:
                    groups[group_val] = {}
                groups[group_val][col] = ref

                # Extract group column name from group_as
                if ref.group_as is not None:
                    if isinstance(ref.group_as, str):
                        group_column_name = ref.group_as
                    elif isinstance(ref.group_as, list) and len(ref.group_as) > 0:
                        group_column_name = ref.group_as[0]

            if group_column_name is None:
                group_column_name = "group"  # Default fallback

            # Build DataFrame chunks for each group value
            for group_val, group_col_refs in sorted(groups.items(), key=lambda x: int(x[0])):
                group_data = {key: np.copy(value) for key, value in metadata_columns.items()}
                group_data["layer"] = np.repeat(layer_name, base_rows)
                # Ensure group value is always string for consistent faceting
                group_data[group_column_name] = np.repeat(str(group_val), base_rows)

                # Add non-group columns (same for all groups)
                for column, ref in non_group_refs.items():
                    group_data[column] = _resolve_field(
                        ref,
                        layer_name,
                        projections,
                        scalars,
                        belief_states,
                        analysis_concat_layers,
                        base_rows,
                        metadata_columns,
                    )

                # Add group-specific columns with base names (strip group index)
                for column, ref in group_col_refs.items():
                    base_col_name = _extract_base_column_name(column, group_val, None)
                    group_data[base_col_name] = _resolve_field(
                        ref,
                        layer_name,
                        projections,
                        scalars,
                        belief_states,
                        analysis_concat_layers,
                        base_rows,
                        metadata_columns,
                    )

                frames.append(pd.DataFrame(group_data))
        else:
            # No group expansion: standard DataFrame construction
            layer_data = {key: np.copy(value) for key, value in metadata_columns.items()}
            layer_data["layer"] = np.repeat(layer_name, base_rows)

            for column, ref in expanded_mappings.items():
                layer_data[column] = _resolve_field(
                    ref,
                    layer_name,
                    projections,
                    scalars,
                    belief_states,
                    analysis_concat_layers,
                    base_rows,
                    metadata_columns,
                )
            frames.append(pd.DataFrame(layer_data))

    return pd.concat(frames, ignore_index=True)


def _build_dataframe(
    viz_cfg: ActivationVisualizationConfig,
    metadata_columns: Mapping[str, Any],
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    scalar_history: Mapping[str, list[tuple[int, float]]],
    scalar_history_step: int | None,
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    layer_names: list[str],
) -> pd.DataFrame:
    """Build a DataFrame from visualization configuration."""
    # Handle combined mappings (multiple data sources with labels)
    if viz_cfg.data_mapping.combined is not None:
        combined_frames: list[pd.DataFrame] = []
        combine_column = viz_cfg.data_mapping.combine_as
        assert combine_column is not None, "combine_as should be validated in config"

        for section in viz_cfg.data_mapping.combined:
            section_df = _build_dataframe_for_mappings(
                section.mappings,
                metadata_columns,
                projections,
                scalars,
                belief_states,
                analysis_concat_layers,
                layer_names,
            )
            section_df[combine_column] = section.label
            combined_frames.append(section_df)

        return pd.concat(combined_frames, ignore_index=True)

    # Check if this is a scalar_pattern or scalar_history visualization
    has_scalar_pattern = any(ref.source == "scalar_pattern" for ref in viz_cfg.data_mapping.mappings.values())
    has_scalar_history = any(ref.source == "scalar_history" for ref in viz_cfg.data_mapping.mappings.values())

    if has_scalar_pattern or has_scalar_history:
        if scalar_history_step is None:
            raise ConfigValidationError(
                "Visualization uses scalar_pattern/scalar_history "
                "source but analyze() was called without the `step` parameter."
            )
        if "analysis" not in metadata_columns:
            raise ConfigValidationError("scalar_pattern/scalar_history requires 'analysis' in metadata_columns.")
        analysis_name = str(metadata_columns["analysis"][0])
        return _build_scalar_dataframe(
            viz_cfg.data_mapping.mappings,
            scalars,
            scalar_history,
            analysis_name,
            scalar_history_step,
        )

    if viz_cfg.data_mapping.scalar_series is not None:
        if "analysis" not in metadata_columns:
            raise ConfigValidationError("scalar_series requires 'analysis' in metadata_columns.")
        analysis_name = str(metadata_columns["analysis"][0])
        return _build_scalar_series_dataframe(
            viz_cfg.data_mapping.scalar_series,
            metadata_columns,
            scalars,
            layer_names,
            analysis_name,
        )

    # Standard mappings mode - delegate to helper
    return _build_dataframe_for_mappings(
        viz_cfg.data_mapping.mappings,
        metadata_columns,
        projections,
        scalars,
        belief_states,
        analysis_concat_layers,
        layer_names,
    )


__all__ = [
    "_build_dataframe",
    "_build_dataframe_for_mappings",
    "_build_metadata_columns",
    "_build_scalar_dataframe",
    "_build_scalar_series_dataframe",
    "_extract_base_column_name",
    "_infer_scalar_series_indices",
    "_scalar_series_metadata",
]
