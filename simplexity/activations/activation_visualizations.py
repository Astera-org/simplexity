"""Helpers for building activation visualizations from analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import re

from simplexity.analysis.pca import compute_weighted_pca
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationControlsConfig,
    ActivationVisualizationFieldRef,
    ActivationVisualizationPreprocessStep,
    ScalarSeriesMapping,
)
from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import PlotConfig


@dataclass
class PreparedMetadata:
    """Metadata derived during activation preprocessing."""

    sequences: list[tuple[int, ...]]
    steps: np.ndarray
    select_last_token: bool


@dataclass
class ActivationVisualizationPayload:
    """Rendered visualization plus auxiliary metadata."""

    analysis: str
    name: str
    backend: str
    figure: Any
    dataframe: pd.DataFrame
    controls: "VisualizationControlsState | None"
    plot_config: PlotConfig


@dataclass
class VisualizationControlDetail:
    """Runtime metadata for a single control."""

    type: str
    field: str
    options: list[Any]
    cumulative: bool | None = None


@dataclass
class VisualizationControlsState:
    """Collection of optional control metadata."""

    slider: VisualizationControlDetail | None = None
    dropdown: VisualizationControlDetail | None = None
    toggle: VisualizationControlDetail | None = None
    accumulate_steps: bool = False


_SCALAR_INDEX_SENTINEL = "__SCALAR_INDEX_SENTINEL__"


def _render_title_template(
    title: str | None,
    title_scalars: dict[str, str] | None,
    scalars: Mapping[str, float],
) -> str | None:
    """Render a title template by substituting scalar values.

    Args:
        title: Title string potentially containing format placeholders like {rmse:.3f}
        title_scalars: Mapping from template variable names to scalar keys
        scalars: Available scalar values

    Returns:
        Rendered title string with scalar values substituted, or None if title is None
    """
    if title is None:
        return None

    if title_scalars is None or not title_scalars:
        return title

    scalar_values = {}
    for var_name, scalar_key in title_scalars.items():
        if scalar_key in scalars:
            scalar_values[var_name] = scalars[scalar_key]
        else:
            raise ConfigValidationError(
                f"Title template references scalar '{scalar_key}' (var: '{var_name}') but it is not available. "
                f"Available scalars: {list(scalars.keys())}"
            )

    try:
        return title.format(**scalar_values)
    except (KeyError, ValueError, IndexError) as e:
        raise ConfigValidationError(
            f"Failed to render title template '{title}' with values {scalar_values}: {e}"
        ) from e


def build_visualization_payloads(
    analysis_name: str,
    viz_cfgs: list[ActivationVisualizationConfig],
    *,
    default_backend: str,
    prepared_metadata: PreparedMetadata,
    weights: np.ndarray,
    belief_states: np.ndarray | None,
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    scalar_history_step: int | None,
    analysis_concat_layers: bool,
    layer_names: list[str],
) -> list[ActivationVisualizationPayload]:
    """Materialize and render the configured visualizations for one analysis."""

    payloads: list[ActivationVisualizationPayload] = []
    metadata_columns = _build_metadata_columns(analysis_name, prepared_metadata, weights)
    for viz_cfg in viz_cfgs:
        dataframe = _build_dataframe(
            viz_cfg,
            metadata_columns,
            projections,
            scalars,
            scalar_history_step,
            belief_states,
            analysis_concat_layers,
            layer_names,
        )
        dataframe = _apply_preprocessing(dataframe, viz_cfg.preprocessing)
        plot_cfg = viz_cfg.resolve_plot_config(default_backend)

        if plot_cfg.guides and plot_cfg.guides.title_scalars:
            plot_cfg.guides.title = _render_title_template(
                plot_cfg.guides.title,
                plot_cfg.guides.title_scalars,
                scalars,
            )

        controls = _build_controls_state(dataframe, viz_cfg.controls)
        backend = plot_cfg.backend
        figure = render_visualization(plot_cfg, dataframe, controls)
        payloads.append(
            ActivationVisualizationPayload(
                analysis=analysis_name,
                name=viz_cfg.name,
                backend=backend,
                figure=figure,
                dataframe=dataframe,
                controls=controls,
                plot_config=plot_cfg,
            )
        )
    return payloads


def render_visualization(
    plot_cfg: PlotConfig,
    dataframe: pd.DataFrame,
    controls: VisualizationControlsState | None,
) -> Any:
    registry = DictDataRegistry({plot_cfg.data.source: dataframe})
    return _render_plot(plot_cfg, registry, controls)


def _render_plot(
    plot_cfg: PlotConfig,
    registry: DictDataRegistry,
    controls: VisualizationControlsState | None,
) -> Any:
    if plot_cfg.backend == "plotly":
        return build_plotly_figure(plot_cfg, registry, controls=controls)
    return build_altair_chart(plot_cfg, registry, controls=controls)


def _build_metadata_columns(
    analysis_name: str,
    metadata: PreparedMetadata,
    weights: np.ndarray,
) -> dict[str, Any]:
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


def _parse_component_spec(component: int | str | None) -> tuple[str, int | None, int | None]:
    """
    Parse component into (type, start, end).

    Returns:
        - ("single", val, None) for int component
        - ("wildcard", None, None) for "*"
        - ("range", start, end) for "start...end"
        - ("none", None, None) for None
    """
    if component is None:
        return ("none", None, None)
    if isinstance(component, int):
        return ("single", component, None)
    if component == "*":
        return ("wildcard", None, None)
    if "..." in component:
        parts = component.split("...")
        if len(parts) != 2:
            raise ConfigValidationError(f"Invalid range: {component}")
        try:
            start, end = int(parts[0]), int(parts[1])
            if start >= end:
                raise ConfigValidationError(f"Range start must be < end: {component}")
            return ("range", start, end)
        except ValueError as e:
            raise ConfigValidationError(f"Invalid range: {component}") from e
    raise ConfigValidationError(f"Unrecognized component pattern: {component}")


def _has_field_pattern(field_name: str) -> bool:
    """Check if field name contains * or range pattern."""
    star_count = field_name.count("*")
    range_count = len(re.findall(r"\d+\.\.\.\d+", field_name))

    if star_count + range_count > 1:
        raise ConfigValidationError(f"Field name cannot have multiple patterns: {field_name}")

    return star_count + range_count == 1


def _get_component_count(
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
) -> int:
    """Get number of components available for expansion."""
    if ref.source == "projections":
        if ref.key is None:
            raise ConfigValidationError("Projection refs require key")
        array = _lookup_projection_array(projections, layer_name, ref.key, analysis_concat_layers)
        np_array = np.asarray(array)
        if np_array.ndim == 1:
            raise ConfigValidationError(
                f"Cannot expand 1D projection '{ref.key}'. Patterns require 2D arrays."
            )
        if np_array.ndim != 2:
            raise ConfigValidationError(f"Projection must be 1D or 2D, got {np_array.ndim}D")
        return np_array.shape[1]

    elif ref.source == "belief_states":
        if belief_states is None:
            raise ConfigValidationError("Belief states not available")
        np_array = np.asarray(belief_states)
        if np_array.ndim != 2:
            raise ConfigValidationError(f"Belief states must be 2D, got {np_array.ndim}D")
        return np_array.shape[1]

    else:
        raise ConfigValidationError(f"Component expansion not supported for source: {ref.source}")


def _expand_scalar_keys(
    field_pattern: str,
    key_pattern: str | None,
    layer_name: str,
    scalars: Mapping[str, float],
) -> dict[str, str]:
    """
    Expand scalar field patterns by matching available scalar keys.

    Returns dict of expanded field_name → scalar_key.
    """

    if key_pattern is None:
        raise ConfigValidationError("Scalar wildcard expansion requires a key pattern")

    has_star = "*" in key_pattern
    has_range = bool(re.search(r"\d+\.\.\.\d+", key_pattern))

    if not has_star and not has_range:
        return {field_pattern: key_pattern}

    if has_star:
        escaped_pattern = re.escape(key_pattern).replace(r"\*", r"(\d+)")
        regex_pattern = re.compile(f"^{escaped_pattern}$")
        indices: list[int] = []
        for scalar_key in scalars.keys():
            match = regex_pattern.match(scalar_key)
            if match:
                try:
                    indices.append(int(match.group(1)))
                except (ValueError, IndexError):
                    continue
        if not indices:
            raise ConfigValidationError(f"No scalar keys found matching pattern '{key_pattern}'")
        indices = sorted(set(indices))
    else:
        range_match = re.search(r"(\d+)\.\.\.(\d+)", key_pattern)
        if not range_match:
            raise ConfigValidationError(f"Invalid range pattern in key '{key_pattern}'")
        start_idx = int(range_match.group(1))
        end_idx = int(range_match.group(2))
        indices = list(range(start_idx, end_idx))

    expanded = {}
    for idx in indices:
        expanded_field = field_pattern.replace("*", str(idx)) if "*" in field_pattern else field_pattern
        if has_range:
            expanded_field = re.sub(r"\d+\.\.\.\d+", str(idx), expanded_field)

        expanded_key = key_pattern.replace("*", str(idx)) if "*" in key_pattern else key_pattern
        if has_range:
            expanded_key = re.sub(r"\d+\.\.\.\d+", str(idx), expanded_key)

        expanded[expanded_field] = expanded_key

    return expanded


def _expand_field_mapping(
    field_name: str,
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
) -> dict[str, ActivationVisualizationFieldRef]:
    """
    Expand pattern-based mapping into concrete mappings.

    Returns dict of expanded field_name → FieldRef with concrete component/key values.
    """
    has_pattern = _has_field_pattern(field_name)

    if ref.source == "scalars":
        has_key_pattern = ref.key is not None and ("*" in ref.key or bool(re.match(r".*\d+\.\.\.\d+.*", ref.key)))

        if has_pattern and not has_key_pattern:
            raise ConfigValidationError(f"Field '{field_name}' has pattern but scalar key has no pattern")
        if has_key_pattern and not has_pattern:
            raise ConfigValidationError(f"Scalar key pattern '{ref.key}' requires field name pattern")

        if not has_pattern:
            return {field_name: ref}

        scalar_expansions = _expand_scalar_keys(field_name, ref.key, layer_name, scalars)
        return {
            field: ActivationVisualizationFieldRef(source="scalars", key=key, component=None, reducer=None)
            for field, key in scalar_expansions.items()
        }

    spec_type, start_idx, end_idx = _parse_component_spec(ref.component)
    needs_expansion = spec_type in ("wildcard", "range")

    if has_pattern and not needs_expansion:
        raise ConfigValidationError(f"Field '{field_name}' has pattern but component is not wildcard/range")
    if needs_expansion and not has_pattern:
        raise ConfigValidationError(f"Component pattern '{ref.component}' requires field name pattern")

    if not needs_expansion:
        return {field_name: ref}

    max_components = _get_component_count(ref, layer_name, projections, belief_states, analysis_concat_layers)

    if spec_type == "wildcard":
        components = list(range(max_components))
    else:
        if end_idx > max_components:
            raise ConfigValidationError(
                f"Range {start_idx}...{end_idx} exceeds available components (max: {max_components})"
            )
        components = list(range(start_idx, end_idx))

    expanded = {}
    for comp_idx in components:
        if "*" in field_name:
            expanded_name = field_name.replace("*", str(comp_idx))
        else:
            expanded_name = re.sub(r"\d+\.\.\.\d+", str(comp_idx), field_name)

        expanded[expanded_name] = ActivationVisualizationFieldRef(
            source=ref.source,
            key=ref.key,
            component=comp_idx,
            reducer=ref.reducer,
        )

    return expanded


def _build_scalar_history_dataframe(
    mappings: dict[str, ActivationVisualizationFieldRef],
    scalars: Mapping[str, float],
    analysis_name: str,
    current_step: int,
) -> pd.DataFrame:
    """Build a long-format DataFrame for scalar visualizations at the current step."""
    rows: list[dict[str, Any]] = []

    for field_name, ref in mappings.items():
        if ref.source != "scalar_history":
            continue

        if ref.key is None:
            raise ConfigValidationError("scalar_history field references must specify a key")

        # Determine which scalar keys this mapping should include
        if "*" in ref.key or re.search(r"\d+\.\.\.\d+", ref.key):
            matched_keys = _expand_scalar_history_pattern(ref.key, scalars.keys(), analysis_name)
        else:
            matched_keys = [ref.key if "/" in ref.key else f"{analysis_name}/{ref.key}"]

        for scalar_key in matched_keys:
            if scalar_key not in scalars:
                continue

            value = scalars[scalar_key]
            rows.append(
                {
                    "step": current_step,
                    "layer": _scalar_history_label(scalar_key),
                    field_name: value,
                    "metric": scalar_key,
                }
            )

    if not rows:
        raise ConfigValidationError(
            "Scalar history visualization could not find any matching scalar values. "
            f"Available keys: {list(scalars.keys())}"
        )

    return pd.DataFrame(rows)


def _expand_scalar_history_pattern(
    pattern: str,
    available_keys: Iterable[str],
    analysis_name: str,
) -> list[str]:
    """Expand wildcard/range pattern against available scalar keys."""

    keys = list(available_keys)
    has_prefixed_keys = any("/" in key for key in keys)
    prefix = f"{analysis_name}/"

    normalized_pattern = pattern
    if "/" not in normalized_pattern and has_prefixed_keys:
        normalized_pattern = f"{analysis_name}/{normalized_pattern}"
    elif "/" in normalized_pattern and not has_prefixed_keys and normalized_pattern.startswith(prefix):
        normalized_pattern = normalized_pattern[len(prefix) :]

    pattern_variants = _expand_scalar_history_ranges(normalized_pattern)
    matched: list[str] = []

    for variant in pattern_variants:
        if "*" in variant:
            escaped = re.escape(variant).replace(r"\*", r"([^/]+)")
            regex = re.compile(f"^{escaped}$")
            matched.extend(key for key in keys if regex.match(key))
        else:
            if variant in keys:
                matched.append(variant)

    unique_matches: list[str] = []
    seen: set[str] = set()
    for key in matched:
        if key not in seen:
            seen.add(key)
            unique_matches.append(key)

    if not unique_matches:
        raise ConfigValidationError(f"No scalar history keys found matching pattern '{pattern}'")

    return sorted(unique_matches)


def _expand_scalar_history_ranges(pattern: str) -> list[str]:
    """Expand numeric range tokens (e.g., 0...4) within a scalar history pattern."""

    range_pattern = re.compile(r"(\d+)\.\.\.(\d+)")
    match = range_pattern.search(pattern)
    if not match:
        return [pattern]

    start_idx = int(match.group(1))
    end_idx = int(match.group(2))
    if start_idx >= end_idx:
        raise ConfigValidationError(f"Invalid range pattern in scalar history key '{pattern}'")

    expanded: list[str] = []
    for idx in range(start_idx, end_idx):
        replaced = range_pattern.sub(str(idx), pattern, count=1)
        expanded.extend(_expand_scalar_history_ranges(replaced))
    return expanded


def _scalar_history_label(full_key: str) -> str:
    """Derive a categorical label for scalar history rows based on the key."""

    suffix = full_key.split("/", 1)[1] if "/" in full_key else full_key
    layer_match = re.search(r"(layer_\d+)", suffix)
    if layer_match:
        return layer_match.group(1)
    return suffix


def _build_dataframe(
    viz_cfg: ActivationVisualizationConfig,
    metadata_columns: Mapping[str, Any],
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    scalar_history_step: int | None,
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    layer_names: list[str],
) -> pd.DataFrame:
    # Check if this is a scalar_history visualization
    has_scalar_history = any(
        ref.source == "scalar_history" for ref in viz_cfg.data_mapping.mappings.values()
    )

    if has_scalar_history:
        if scalar_history_step is None:
            raise ConfigValidationError(
                "Visualization uses scalar_history source but analyze() was called without the `step` parameter."
            )
        # Extract analysis name from metadata
        analysis_name = str(metadata_columns.get("analysis", ["unknown"])[0])
        return _build_scalar_history_dataframe(
            viz_cfg.data_mapping.mappings,
            scalars,
            analysis_name,
            scalar_history_step,
        )

    if viz_cfg.data_mapping.scalar_series is not None:
        return _build_scalar_series_dataframe(
            viz_cfg.data_mapping.scalar_series,
            metadata_columns,
            scalars,
            layer_names,
        )
    base_rows = len(metadata_columns["step"])
    frames: list[pd.DataFrame] = []
    for layer_name in layer_names:
        layer_data = {key: np.copy(value) for key, value in metadata_columns.items()}
        layer_data["layer"] = np.repeat(layer_name, base_rows)

        expanded_mappings: dict[str, ActivationVisualizationFieldRef] = {}
        for field_name, ref in viz_cfg.data_mapping.mappings.items():
            try:
                expanded = _expand_field_mapping(
                    field_name, ref, layer_name, projections, scalars, belief_states, analysis_concat_layers
                )
                expanded_mappings.update(expanded)
            except ConfigValidationError as e:
                raise ConfigValidationError(
                    f"Error expanding '{field_name}' for layer '{layer_name}': {e}"
                ) from e

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


def _resolve_field(
    ref: ActivationVisualizationFieldRef,
    layer_name: str,
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    num_rows: int,
    metadata_columns: Mapping[str, Any],
) -> np.ndarray:
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


def _build_scalar_series_dataframe(
    mapping: ScalarSeriesMapping,
    metadata_columns: Mapping[str, Any],
    scalars: Mapping[str, float],
    layer_names: list[str],
) -> pd.DataFrame:
    base_metadata = _scalar_series_metadata(metadata_columns)
    rows: list[dict[str, Any]] = []
    for layer_name in layer_names:
        index_values = mapping.index_values or _infer_scalar_series_indices(mapping, scalars, layer_name)
        for index_value in index_values:
            scalar_key = mapping.key_template.format(layer=layer_name, index=index_value)
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
) -> list[int]:
    template = mapping.key_template.format(layer=layer_name, index=_SCALAR_INDEX_SENTINEL)
    if _SCALAR_INDEX_SENTINEL not in template:
        raise ConfigValidationError(
            "scalar_series.key_template must include '{index}' placeholder to infer index values."
        )
    prefix, suffix = template.split(_SCALAR_INDEX_SENTINEL, 1)
    inferred: set[int] = set()
    for key in scalars.keys():
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
            f"Scalar series could not infer indices for layer '{layer_name}' using key_template '{mapping.key_template}'."
        )
    return sorted(inferred)


def _scalar_series_metadata(metadata_columns: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in metadata_columns.items():
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            metadata[key] = value.flat[0]
        else:
            metadata[key] = value
    return metadata


def _lookup_projection_array(
    projections: Mapping[str, np.ndarray], layer_name: str, key: str | None, concat_layers: bool
) -> np.ndarray:
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


def _lookup_scalar_value(
    scalars: Mapping[str, float], layer_name: str, key: str, concat_layers: bool
) -> float:
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
    np_array = np.asarray(belief_states)
    if ref.reducer == "argmax":
        return np.argmax(np_array, axis=1)
    if ref.reducer == "l2_norm":
        return np.linalg.norm(np_array, axis=1)
    component = ref.component if ref.component is not None else 0
    if component < 0 or component >= np_array.shape[1]:
        raise ConfigValidationError(
            f"Belief state component {component} is out of bounds for dimension {np_array.shape[1]}"
        )
    return np_array[:, component]


def _expand_preprocessing_fields(field_patterns: list[str], available_columns: list[str]) -> list[str]:
    """
    Expand wildcard and range patterns in preprocessing field lists.

    Args:
        field_patterns: List of field names, may contain patterns like "belief_*" or "prob_0...3"
        available_columns: List of column names available in the DataFrame

    Returns:
        Expanded list of field names with patterns replaced by matching columns
    """

    expanded: list[str] = []
    for pattern in field_patterns:
        # Check if this is a pattern
        if "*" in pattern or re.search(r"\d+\.\.\.\d+", pattern):
            # Extract the numeric pattern if it's a range
            range_match = re.search(r"(\d+)\.\.\.(\d+)", pattern)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                component_range = list(range(start, end))
                # Replace range pattern with each index
                for idx in component_range:
                    expanded_name = re.sub(r"\d+\.\.\.\d+", str(idx), pattern)
                    if expanded_name in available_columns:
                        expanded.append(expanded_name)
                    else:
                        raise ConfigValidationError(
                            f"Preprocessing pattern '{pattern}' expanded to '{expanded_name}' but column not found in DataFrame. "
                            f"Available columns: {', '.join(sorted(available_columns))}"
                        )
            elif "*" in pattern:
                # Wildcard pattern - find all matching columns
                regex_pattern = pattern.replace("*", r"(\d+)")
                regex = re.compile(f"^{regex_pattern}$")
                matches = []
                for col in available_columns:
                    match = regex.match(col)
                    if match:
                        # Extract the numeric part for sorting
                        try:
                            idx = int(match.group(1))
                            matches.append((idx, col))
                        except (IndexError, ValueError):
                            continue
                if not matches:
                    raise ConfigValidationError(
                        f"Preprocessing pattern '{pattern}' did not match any columns in DataFrame. "
                        f"Available columns: {', '.join(sorted(available_columns))}"
                    )
                # Sort by index and add column names
                matches.sort(key=lambda x: x[0])
                expanded.extend([col for _, col in matches])
            else:
                raise ConfigValidationError(f"Invalid preprocessing field pattern: {pattern}")
        else:
            # Not a pattern, just add as-is
            expanded.append(pattern)

    return expanded


def _apply_preprocessing(
    dataframe: pd.DataFrame, steps: list[ActivationVisualizationPreprocessStep]
) -> pd.DataFrame:
    result = dataframe.copy()
    available_columns = list(result.columns)

    for step in steps:
        # Validate output_fields don't contain patterns
        for output_field in step.output_fields:
            if "*" in output_field or "..." in output_field:
                raise ConfigValidationError(
                    f"Preprocessing output_fields cannot contain patterns. Found: '{output_field}'"
                )

        # Expand input_fields patterns
        expanded_input_fields = _expand_preprocessing_fields(step.input_fields, available_columns)

        # Create a modified step with expanded fields
        expanded_step = ActivationVisualizationPreprocessStep(
            type=step.type, input_fields=expanded_input_fields, output_fields=step.output_fields
        )

        if step.type == "project_to_simplex":
            result = _project_to_simplex(result, expanded_step)
        elif step.type == "combine_rgb":
            result = _combine_rgb(result, expanded_step)
        else:  # pragma: no cover - defensive for future types
            raise ConfigValidationError(f"Unsupported preprocessing op '{step.type}'")

        # Update available columns for next step
        available_columns = list(result.columns)

    return result


def _project_to_simplex(
    dataframe: pd.DataFrame, step: ActivationVisualizationPreprocessStep
) -> pd.DataFrame:
    required = step.input_fields
    for column in required:
        if column not in dataframe:
            raise ConfigValidationError(
                f"Preprocessing step requires column '{column}' but it is missing from the dataframe."
            )
    p0, p1, p2 = (dataframe[col].astype(float) for col in required)
    x = p1 + 0.5 * p2
    y = (np.sqrt(3.0) / 2.0) * p2
    dataframe[step.output_fields[0]] = x
    dataframe[step.output_fields[1]] = y
    return dataframe


def _combine_rgb(dataframe: pd.DataFrame, step: ActivationVisualizationPreprocessStep) -> pd.DataFrame:
    # ---- Validation ----
    # Note: input_fields have already been expanded by _expand_preprocessing_fields()
    # at this point, so we just validate the expanded result
    if len(step.output_fields) != 1:
        raise ConfigValidationError("combine_rgb requires exactly one output_field.")
    if len(step.input_fields) < 3:
        raise ConfigValidationError("combine_rgb requires at least three input_fields.")

    # Make sure all input columns exist
    for field in step.input_fields:
        if field not in dataframe:
            raise ConfigValidationError(
                f"combine_rgb requires column '{field}' but it is missing from the dataframe."
            )

    def _channel_to_int(series: pd.Series) -> pd.Series:
        return (series.clip(0.0, 1.0) * 255).round().astype(int)

    # ---- Case 1: exactly 3 inputs -> keep old behavior ----
    if len(step.input_fields) == 3:
        r, g, b = step.input_fields
        r_vals = _channel_to_int(dataframe[r])
        g_vals = _channel_to_int(dataframe[g])
        b_vals = _channel_to_int(dataframe[b])

    # ---- Case 2: >3 inputs -> PCA to 3D, then map to RGB ----
    else:
        import jax.numpy as jnp
        import numpy as np

        # Stack the selected columns into an (n_samples, n_features) matrix
        X_np = dataframe[step.input_fields].to_numpy(dtype=float)
        X_jax = jnp.asarray(X_np)

        # Unweighted PCA (weights=None) to up to 3 components
        # We pass n_components=3, but compute_weighted_pca will cap it at min(n_samples, n_features)
        # via its own logic if you change it to allow that, or you can just pass None and slice.
        pca_res = compute_weighted_pca(
            X_jax,
            n_components=None,   # let it pick max_rank
            weights=None,
            center=True,
        )

        # Get projected coordinates, shape: (n_samples, k) where k = max_rank
        proj = np.asarray(pca_res["X_proj"])  # convert from jax.Array to numpy

        # Ensure we have 3 channels: take first 3 components, pad with zeros if fewer
        if proj.shape[1] >= 3:
            proj3 = proj[:, :3]
        else:
            # This is rare (happens when n_samples < 3). Pad extra dims with zeros.
            pad_width = 3 - proj.shape[1]
            proj3 = np.pad(proj, ((0, 0), (0, pad_width)), mode="constant")

        # Min-max normalize each component to [0, 1] across the dataset
        mins = proj3.min(axis=0)
        maxs = proj3.max(axis=0)
        ranges = maxs - mins
        # Avoid divide-by-zero: if range is 0, just leave that channel at 0.5
        ranges_safe = np.where(ranges > 0, ranges, 1.0)
        colors = (proj3 - mins) / ranges_safe
        colors[:, ranges == 0] = 0.5

        colors = np.clip(colors, 0.0, 1.0)

        # Turn into Series so we can reuse _channel_to_int
        r_vals = _channel_to_int(pd.Series(colors[:, 0], index=dataframe.index))
        g_vals = _channel_to_int(pd.Series(colors[:, 1], index=dataframe.index))
        b_vals = _channel_to_int(pd.Series(colors[:, 2], index=dataframe.index))

    # ---- Build hex color column ----
    dataframe[step.output_fields[0]] = [
        f"#{rv:02x}{gv:02x}{bv:02x}"
        for rv, gv, bv in zip(r_vals, g_vals, b_vals)
    ]
    return dataframe


def _build_controls_state(
    dataframe: pd.DataFrame, controls_cfg: ActivationVisualizationControlsConfig | None
) -> VisualizationControlsState | None:
    if controls_cfg is None:
        return None
    slider = _build_control_detail(dataframe, "slider", controls_cfg.slider, controls_cfg.cumulative)
    dropdown = _build_control_detail(dataframe, "dropdown", controls_cfg.dropdown)
    toggle = _build_control_detail(dataframe, "toggle", controls_cfg.toggle)
    return VisualizationControlsState(
        slider=slider,
        dropdown=dropdown,
        toggle=toggle,
        accumulate_steps=controls_cfg.accumulate_steps,
    )


def _build_control_detail(
    dataframe: pd.DataFrame,
    control_type: str,
    field: str | None,
    cumulative: bool | None = None,
) -> VisualizationControlDetail | None:
    if field is None:
        return None
    if field not in dataframe:
        raise ConfigValidationError(f"Control field '{field}' is not present in visualization dataframe.")
    options = list(pd.unique(dataframe[field]))
    return VisualizationControlDetail(type=control_type, field=field, options=options, cumulative=cumulative)


__all__ = [
    "ActivationVisualizationPayload",
    "PreparedMetadata",
    "VisualizationControlDetail",
    "VisualizationControlsState",
    "build_visualization_payloads",
    "render_visualization",
]
