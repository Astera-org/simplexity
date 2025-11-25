"""Helpers for building activation visualizations from analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

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
            belief_states,
            analysis_concat_layers,
            layer_names,
        )
        dataframe = _apply_preprocessing(dataframe, viz_cfg.preprocessing)
        plot_cfg = viz_cfg.resolve_plot_config(default_backend)
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


def _build_dataframe(
    viz_cfg: ActivationVisualizationConfig,
    metadata_columns: Mapping[str, Any],
    projections: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    belief_states: np.ndarray | None,
    analysis_concat_layers: bool,
    layer_names: list[str],
) -> pd.DataFrame:
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
        for column, ref in viz_cfg.data_mapping.mappings.items():
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


def _apply_preprocessing(
    dataframe: pd.DataFrame, steps: list[ActivationVisualizationPreprocessStep]
) -> pd.DataFrame:
    result = dataframe.copy()
    for step in steps:
        if step.type == "project_to_simplex":
            result = _project_to_simplex(result, step)
        elif step.type == "combine_rgb":
            result = _combine_rgb(result, step)
        else:  # pragma: no cover - defensive for future types
            raise ConfigValidationError(f"Unsupported preprocessing op '{step.type}'")
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
    if len(step.input_fields) != 3 or len(step.output_fields) != 1:
        raise ConfigValidationError("combine_rgb requires three input_fields and one output_field.")
    r, g, b = step.input_fields
    for field in (r, g, b):
        if field not in dataframe:
            raise ConfigValidationError(
                f"combine_rgb requires column '{field}' but it is missing from the dataframe."
            )

    def _channel_to_int(series: pd.Series) -> pd.Series:
        return (series.clip(0.0, 1.0) * 255).round().astype(int)

    r_vals = _channel_to_int(dataframe[r])
    g_vals = _channel_to_int(dataframe[g])
    b_vals = _channel_to_int(dataframe[b])
    dataframe[step.output_fields[0]] = [f"#{rv:02x}{gv:02x}{bv:02x}" for rv, gv, bv in zip(r_vals, g_vals, b_vals)]
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
