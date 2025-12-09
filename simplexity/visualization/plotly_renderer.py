"""Plotly renderer for visualization PlotConfigs."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as qualitative_colors

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_pipeline import (
    build_plot_level_dataframe,
    resolve_layer_dataframe,
)
from simplexity.visualization.data_registry import DataRegistry
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)

LOGGER = logging.getLogger(__name__)

_HEX_COLOR_PATTERN = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def build_plotly_figure(
    plot_cfg: PlotConfig,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
    controls: Any | None = None,
):
    """Render a PlotConfig into a Plotly Figure (currently 3D scatter only)."""
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Plotly rendering.")
    if len(plot_cfg.layers) != 1:
        raise ConfigValidationError("Plotly renderer currently supports exactly one layer.")

    layer = plot_cfg.layers[0]
    if layer.geometry.type != "point":
        raise ConfigValidationError("Plotly renderer currently supports point geometry.")

    plot_df = build_plot_level_dataframe(plot_cfg.data, plot_cfg.transforms, data_registry)
    layer_df = resolve_layer_dataframe(layer, plot_df, data_registry)

    has_z = bool(layer.aesthetics and layer.aesthetics.z and layer.aesthetics.z.field)
    if has_z:
        figure = _build_scatter3d(layer, layer_df, controls)
    else:
        figure = _build_scatter2d(layer, layer_df, controls)
    figure = _apply_plot_level_properties(figure, plot_cfg.guides, plot_cfg.size, plot_cfg.background, layer.aesthetics)
    return figure


def _build_scatter3d(layer: LayerConfig, df: pd.DataFrame, controls: Any | None):
    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")
    z_field = _require_field(aes.z, "z")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    dropdown = _resolve_layer_dropdown(df, controls)
    slider_enabled = not (getattr(controls, "accumulate_steps", False))
    slider = _resolve_slider_control(df, controls if slider_enabled else None)
    color_map = _build_color_discrete_map(df, color_field, aes.color)
    color_specs = _build_color_group_specs(df, color_field, aes.color, color_map)

    if slider:
        figure = _build_slider_scatter3d(
            df,
            slider,
            dropdown,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            aes,
            layer,
        )
    elif dropdown:
        figure = _build_layer_filtered_scatter3d(
            df,
            dropdown,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            aes,
            layer,
        )
    else:
        traces = _scatter3d_traces(
            df,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer.name,
        )
        figure = go.Figure(data=traces)
        figure = _apply_constant_channels(figure, aes)
        _maybe_update_trace_name(figure, layer, color_field)

    _apply_legend_visibility(figure, aes)

    return figure


def _build_scatter2d(layer: LayerConfig, df: pd.DataFrame, controls: Any | None):
    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    dropdown = _resolve_layer_dropdown(df, controls)
    slider_enabled = not (getattr(controls, "accumulate_steps", False))
    slider = _resolve_slider_control(df, controls if slider_enabled else None)
    color_map = _build_color_discrete_map(df, color_field, aes.color)
    color_specs = _build_color_group_specs(df, color_field, aes.color, color_map)

    if slider:
        figure = _build_slider_scatter2d(
            df,
            slider,
            dropdown,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            aes,
            layer,
        )
    elif dropdown:
        figure = _build_layer_filtered_scatter2d(
            df,
            dropdown,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            aes,
            layer,
        )
    else:
        traces = _scatter2d_traces(
            df,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer.name,
        )
        figure = go.Figure(data=traces)
        figure = _apply_constant_channels(figure, aes)
        _maybe_update_trace_name(figure, layer, color_field)

    _apply_legend_visibility(figure, aes)
    return figure


def _apply_plot_level_properties(
    figure,
    guides: PlotLevelGuideConfig,
    size: PlotSizeConfig,
    background: str | None,
    aes: AestheticsConfig,
):
    title_lines = [guides.title] if guides.title else []
    title_lines += [text for text in (guides.subtitle, guides.caption) if text]
    if title_lines:
        figure.update_layout(title="<br>".join(title_lines))
    if size.width or size.height:
        figure.update_layout(width=size.width, height=size.height)

    has_3d = any(trace.type == "scatter3d" for trace in figure.data)
    x_title = _axis_title(aes.x)
    y_title = _axis_title(aes.y)
    z_title = _axis_title(aes.z)
    if has_3d:
        scene_updates: dict[str, Any] = {}
        if x_title:
            scene_updates.setdefault("xaxis", {})["title"] = x_title
        if y_title:
            scene_updates.setdefault("yaxis", {})["title"] = y_title
        if z_title:
            scene_updates.setdefault("zaxis", {})["title"] = z_title
        if background:
            scene_updates["bgcolor"] = background
        if scene_updates:
            figure.update_layout(scene=scene_updates)
    else:
        axis_updates: dict[str, Any] = {}
        if x_title:
            axis_updates.setdefault("xaxis", {})["title"] = x_title
        if y_title:
            axis_updates.setdefault("yaxis", {})["title"] = y_title
        if background:
            axis_updates["plot_bgcolor"] = background
        if axis_updates:
            figure.update_layout(**axis_updates)

    if guides.labels:
        LOGGER.info("Plot-level labels are not yet implemented for Plotly; skipping %s labels.", len(guides.labels))
    return figure


def _require_field(channel: ChannelAestheticsConfig | None, name: str) -> str:
    if channel is None or not channel.field:
        raise ConfigValidationError(f"Plotly renderer requires '{name}' channel with a field specified.")
    return channel.field


def _optional_field(channel: ChannelAestheticsConfig | None) -> str | None:
    if channel is None:
        return None
    return channel.field


def _collect_tooltip_fields(tooltips: list[ChannelAestheticsConfig] | None) -> list[str]:
    if not tooltips:
        return []
    fields: list[str] = []
    for tooltip in tooltips:
        if tooltip.field is None:
            raise ConfigValidationError("Plotly renderer tooltip entries must reference a data field.")
        fields.append(tooltip.field)
    return fields


def _resolve_opacity(channel: ChannelAestheticsConfig | None) -> float | None:
    if channel is None:
        return None
    if channel.value is None:
        raise ConfigValidationError("Plotly renderer opacity channel must specify a constant value.")
    try:
        opacity = float(channel.value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError("Opacity channel must be a numeric constant.") from exc
    if not 0.0 <= opacity <= 1.0:
        raise ConfigValidationError("Opacity value must be between 0 and 1.")
    return opacity


def _axis_title(channel: ChannelAestheticsConfig | None) -> str | None:
    if channel is None:
        return None
    return channel.title or channel.field


def _apply_constant_channels(figure, aes: AestheticsConfig):
    if aes.color and aes.color.value is not None:
        figure.update_traces(marker={"color": aes.color.value}, selector={"type": "scatter3d"})
        figure.update_traces(marker={"color": aes.color.value}, selector={"type": "scatter"})
        for frame in getattr(figure, "frames", []) or []:
            for trace in frame.data:
                if hasattr(trace, "marker"):
                    trace.marker = trace.marker or {}
                    trace.marker["color"] = aes.color.value
    if aes.size and aes.size.value is not None:
        figure.update_traces(marker={"size": aes.size.value}, selector={"type": "scatter3d"})
        figure.update_traces(marker={"size": aes.size.value}, selector={"type": "scatter"})
        for frame in getattr(figure, "frames", []) or []:
            for trace in frame.data:
                if hasattr(trace, "marker"):
                    trace.marker = trace.marker or {}
                    trace.marker["size"] = aes.size.value
    return figure


def _apply_legend_visibility(figure, aes: AestheticsConfig):
    if not _legend_hidden(aes.color):
        return
    for trace in figure.data:
        trace.showlegend = False
    for frame in getattr(figure, "frames", []) or []:
        for trace in frame.data:
            trace.showlegend = False


def _legend_hidden(color_cfg: ChannelAestheticsConfig | None) -> bool:
    return bool(color_cfg and color_cfg.legend and color_cfg.legend.visible is False)


def _maybe_update_trace_name(figure, layer: LayerConfig, color_field: str | None):
    if len(figure.data) != 1:
        return
    trace_name = layer.name or (color_field or "3d_scatter")
    figure.update_traces(name=trace_name, selector={"type": "scatter3d"})


def _resolve_layer_dropdown(df: pd.DataFrame, controls: Any | None) -> tuple[str, list[Any]] | None:
    if not controls:
        return None
    dropdown = getattr(controls, "dropdown", None)
    field_name = getattr(dropdown, "field", None) if dropdown else None
    if field_name != "layer" or field_name not in df.columns:
        return None
    raw_options = getattr(dropdown, "options", None) or []
    options = [_normalize_option(value) for value in raw_options]
    valid_values = [value for value in options if value in set(df[field_name])]
    if len(valid_values) <= 1:
        return None
    return field_name, valid_values


def _resolve_slider_control(df: pd.DataFrame, controls: Any | None) -> tuple[str, list[Any]] | None:
    if not controls:
        return None
    slider = getattr(controls, "slider", None)
    field_name = getattr(slider, "field", None) if slider else None
    if field_name is None or field_name not in df.columns:
        return None
    raw_options = getattr(slider, "options", None)
    option_values = raw_options or list(pd.unique(df[field_name]))
    options = [_normalize_option(value) for value in option_values]
    if len(options) <= 1:
        return None
    # Preserve order if numeric; otherwise keep as strings
    try:
        options = sorted(set(options), key=lambda v: float(v))
    except (TypeError, ValueError):  # pragma: no cover - fallback for non-numeric
        options = sorted(dict.fromkeys(options))
    return field_name, options


def _build_layer_filtered_scatter3d(
    df: pd.DataFrame,
    dropdown: tuple[str, list[Any]],
    x_field: str,
    y_field: str,
    z_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    aes: AestheticsConfig,
    layer: LayerConfig,
):
    field_name, options = dropdown
    traces: list[Any] = []
    trace_ranges: list[tuple[int, int]] = []
    available: list[Any] = []

    for option in options:
        subset = df[df[field_name] == option]
        assert isinstance(subset, pd.DataFrame)
        if subset.empty:
            continue
        layer_index = len(available)
        available.append(option)
        subset_traces = _scatter3d_traces(
            subset,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=str(option),
        )
        for trace in subset_traces:
            trace.visible = layer_index == 0
        start = len(traces)
        traces.extend(subset_traces)
        trace_ranges.append((start, len(traces)))

    if len(available) <= 1:
        traces = _scatter3d_traces(
            df,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer.name,
        )
        figure = go.Figure(data=traces)
        figure = _apply_constant_channels(figure, aes)
        _maybe_update_trace_name(figure, layer, color_field)
        return figure

    figure = go.Figure(data=traces)
    figure = _apply_constant_channels(figure, aes)
    _add_layer_dropdown_menu(figure, available, trace_ranges)
    _maybe_update_trace_name(figure, layer, color_field)
    return figure


def _build_layer_filtered_scatter2d(
    df: pd.DataFrame,
    dropdown: tuple[str, list[Any]],
    x_field: str,
    y_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    aes: AestheticsConfig,
    layer: LayerConfig,
):
    field_name, options = dropdown
    traces: list[Any] = []
    trace_ranges: list[tuple[int, int]] = []
    available: list[Any] = []

    for option in options:
        subset = df[df[field_name] == option]
        assert isinstance(subset, pd.DataFrame)
        if subset.empty:
            continue
        layer_index = len(available)
        available.append(option)
        subset_traces = _scatter2d_traces(
            subset,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=str(option),
        )
        for trace in subset_traces:
            trace.visible = layer_index == 0
        start = len(traces)
        traces.extend(subset_traces)
        trace_ranges.append((start, len(traces)))

    if len(available) <= 1:
        traces = _scatter2d_traces(
            df,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer.name,
        )
        figure = go.Figure(data=traces)
        figure = _apply_constant_channels(figure, aes)
        _maybe_update_trace_name(figure, layer, color_field)
        return figure

    figure = go.Figure(data=traces)
    figure = _apply_constant_channels(figure, aes)
    _add_layer_dropdown_menu(figure, available, trace_ranges)
    _maybe_update_trace_name(figure, layer, color_field)
    return figure


def _build_slider_scatter3d(
    df: pd.DataFrame,
    slider: tuple[str, list[Any]],
    dropdown: tuple[str, list[Any]] | None,
    x_field: str,
    y_field: str,
    z_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    aes: AestheticsConfig,
    layer: LayerConfig,
):
    slider_field, slider_values = slider
    layer_field = dropdown[0] if dropdown else None
    layer_options = dropdown[1] if dropdown else [None]

    traces: list[Any] = []
    trace_ranges: list[tuple[int, int]] = []
    available_layers: list[Any] = []
    frames_by_value: dict[str, list[Any]] = {str(value): [] for value in slider_values}

    for option in layer_options:
        subset = df if option is None else df[df[layer_field] == option]
        if subset.empty:
            continue
        layer_index = len(available_layers)
        available_layers.append(option)
        layer_label = str(option) if option is not None else layer.name

        initial_subset = subset[subset[slider_field] == slider_values[0]]
        assert isinstance(initial_subset, pd.DataFrame)
        subset_traces = _scatter3d_traces(
            initial_subset,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer_label,
            keep_empty=True,
        )
        if dropdown:
            for trace in subset_traces:
                trace.visible = layer_index == 0
        start = len(traces)
        traces.extend(subset_traces)
        trace_ranges.append((start, len(traces)))

        for slider_value in slider_values:
            slider_subset = subset[subset[slider_field] == slider_value]
            assert isinstance(slider_subset, pd.DataFrame)
            frame_traces = _scatter3d_traces(
                slider_subset,
                x_field,
                y_field,
                z_field,
                color_field,
                size_field,
                hover_fields,
                opacity_value,
                color_specs,
                layer_name=layer_label,
                keep_empty=True,
            )
            frames_by_value[str(slider_value)].extend(frame_traces)

    figure = go.Figure(data=traces)
    figure.frames = _build_slider_frames(frames_by_value, slider_values)
    _add_slider_layout(figure, slider_field, slider_values)

    if dropdown and available_layers:
        _add_layer_dropdown_menu(figure, available_layers, trace_ranges)
    else:
        _maybe_update_trace_name(figure, layer, color_field)

    figure = _apply_constant_channels(figure, aes)
    return figure


def _build_slider_scatter2d(
    df: pd.DataFrame,
    slider: tuple[str, list[Any]],
    dropdown: tuple[str, list[Any]] | None,
    x_field: str,
    y_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    aes: AestheticsConfig,
    layer: LayerConfig,
):
    slider_field, slider_values = slider
    layer_field = dropdown[0] if dropdown else None
    layer_options = dropdown[1] if dropdown else [None]

    traces: list[Any] = []
    trace_ranges: list[tuple[int, int]] = []
    available_layers: list[Any] = []
    frames_by_value: dict[str, list[Any]] = {str(value): [] for value in slider_values}

    for option in layer_options:
        subset = df if option is None else df[df[layer_field] == option]
        if subset.empty:
            continue
        layer_index = len(available_layers)
        available_layers.append(option)
        layer_label = str(option) if option is not None else layer.name

        initial_subset = subset[subset[slider_field] == slider_values[0]]
        assert isinstance(initial_subset, pd.DataFrame)
        subset_traces = _scatter2d_traces(
            initial_subset,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            layer_name=layer_label,
            keep_empty=True,
        )
        for trace in subset_traces:
            trace.visible = layer_index == 0
        start = len(traces)
        traces.extend(subset_traces)
        trace_ranges.append((start, len(traces)))

        for slider_value in slider_values:
            slider_subset = subset[subset[slider_field] == slider_value]
            assert isinstance(slider_subset, pd.DataFrame)
            frame_traces = _scatter2d_traces(
                slider_subset,
                x_field,
                y_field,
                color_field,
                size_field,
                hover_fields,
                opacity_value,
                color_specs,
                layer_name=layer_label,
                keep_empty=True,
            )
            frames_by_value[str(slider_value)].extend(frame_traces)

    figure = go.Figure(data=traces)
    figure.frames = _build_slider_frames(frames_by_value, slider_values)
    _add_slider_layout(figure, slider_field, slider_values)

    if dropdown and available_layers:
        _add_layer_dropdown_menu(figure, available_layers, trace_ranges)
    else:
        _maybe_update_trace_name(figure, layer, color_field)

    figure = _apply_constant_channels(figure, aes)
    return figure


def _add_layer_dropdown_menu(
    figure,
    options: list[Any],
    trace_ranges: list[tuple[int, int]],
):
    total_traces = len(figure.data)
    buttons = []
    for option, (start, end) in zip(options, trace_ranges, strict=False):
        visible = [False] * total_traces
        for idx in range(start, end):
            visible[idx] = True
        buttons.append(
            {
                "label": str(option),
                "method": "update",
                "args": [{"visible": visible}],
            }
        )

    figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.05,
                "xanchor": "left",
                "y": 1,
                "yanchor": "top",
                "pad": {"l": 10, "r": 10, "t": 0, "b": 0},
            }
        ]
    )


def _build_slider_frames(frames_by_value: dict[str, list[Any]], slider_values: list[Any]):
    frames: list[go.Frame] = []
    for value in slider_values:
        name = str(value)
        frame_traces = frames_by_value.get(name, [])
        frames.append(go.Frame(name=name, data=frame_traces))
    return frames


def _add_slider_layout(figure, field_name: str, slider_values: list[Any]):
    if not slider_values:
        return
    steps = []
    for value in slider_values:
        label = str(value)
        steps.append(
            {
                "label": label,
                "method": "animate",
                "args": [
                    [label],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            }
        )
    figure.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": f"{field_name}="},
                "pad": {"t": 40, "b": 0},
                "steps": steps,
            }
        ]
    )


@dataclass
class ColorGroupSpec:
    """Specification for a color grouping in Plotly rendering."""
    label: str | None
    value: Any | None
    constant_color: str | None
    mode: Literal["none", "literal", "discrete", "field"] = "none"


def _scatter3d_traces(
    df: pd.DataFrame,
    x_field: str,
    y_field: str,
    z_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    *,
    layer_name: str | None = None,
    keep_empty: bool = False,
) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    for idx, spec in enumerate(color_specs):
        subset = _subset_for_spec(df, color_field, spec)
        if subset.empty and not keep_empty:
            continue
        marker = _build_marker(subset, color_field, size_field, spec)
        customdata = _build_customdata(subset, hover_fields)
        trace = go.Scatter3d(
            x=subset[x_field].tolist(),
            y=subset[y_field].tolist(),
            z=subset[z_field].tolist(),
            mode="markers",
            name=_derive_trace_name(layer_name, spec, idx),
            marker=marker,
            customdata=customdata,
            hovertemplate=_build_hovertemplate(hover_fields),
        )
        if spec.mode == "literal":
            trace.showlegend = False
        if opacity_value is not None:
            trace.opacity = opacity_value
        traces.append(trace)
    if not traces:
        empty_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", name=layer_name or "scatter3d")
        if opacity_value is not None:
            empty_trace.opacity = opacity_value
        traces.append(empty_trace)
    return traces


def _scatter2d_traces(
    df: pd.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    *,
    layer_name: str | None = None,
    keep_empty: bool = False,
) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    for idx, spec in enumerate(color_specs):
        subset = _subset_for_spec(df, color_field, spec)
        if subset.empty and not keep_empty:
            continue
        marker = _build_marker(subset, color_field, size_field, spec)
        customdata = _build_customdata(subset, hover_fields)
        trace = go.Scatter(
            x=subset[x_field].tolist(),
            y=subset[y_field].tolist(),
            mode="markers",
            name=_derive_trace_name(layer_name, spec, idx),
            marker=marker,
            customdata=customdata,
            hovertemplate=_build_hovertemplate(hover_fields),
        )
        if spec.mode == "literal":
            trace.showlegend = False
        if opacity_value is not None:
            trace.opacity = opacity_value
        traces.append(trace)
    if not traces:
        empty_trace = go.Scatter(x=[], y=[], mode="markers", name=layer_name or "scatter")
        if opacity_value is not None:
            empty_trace.opacity = opacity_value
        traces.append(empty_trace)
    return traces


def _subset_for_spec(df: pd.DataFrame, color_field: str | None, spec: ColorGroupSpec) -> pd.DataFrame:
    if spec.mode != "discrete" or color_field is None:
        return df
    result = df[df[color_field] == spec.value]
    assert isinstance(result, pd.DataFrame)
    return result


def _build_marker(
    df: pd.DataFrame,
    color_field: str | None,
    size_field: str | None,
    spec: ColorGroupSpec,
) -> dict[str, Any]:
    marker: dict[str, Any] = {}
    if size_field and size_field in df.columns:
        marker["size"] = df[size_field].tolist()
    if spec.mode == "literal" and color_field and color_field in df.columns:
        marker["color"] = df[color_field].tolist()
    elif spec.mode == "discrete" and spec.constant_color is not None:
        marker["color"] = spec.constant_color
    elif spec.mode == "field" and color_field and color_field in df.columns:
        marker["color"] = df[color_field].tolist()
    return marker


def _build_customdata(df: pd.DataFrame, hover_fields: list[str]) -> Any:
    if not hover_fields:
        return None
    missing = [field for field in hover_fields if field not in df.columns]
    if missing:
        raise ConfigValidationError(f"Tooltip field(s) {missing} are missing from dataframe.")
    return df[hover_fields].to_numpy()


def _build_hovertemplate(hover_fields: list[str]) -> str | None:
    if not hover_fields:
        return None
    template_parts = [f"{field}: %{{customdata[{idx}]}}" for idx, field in enumerate(hover_fields)]
    return "<br>".join(template_parts) + "<extra></extra>"


def _derive_trace_name(layer_name: str | None, spec: ColorGroupSpec, idx: int) -> str:
    if spec.label and layer_name:
        return f"{layer_name} - {spec.label}"
    if spec.label:
        return spec.label
    if layer_name:
        return layer_name
    return f"series_{idx + 1}"


def _build_color_discrete_map(
    df: pd.DataFrame, color_field: str | None, color_cfg: ChannelAestheticsConfig | None
) -> dict[str, str] | None:
    if color_field is None or color_cfg is None:
        return None
    if color_cfg.type not in {"nominal", "ordinal"}:
        return None
    if color_field not in df.columns:
        return None
    series = df[color_field]
    assert isinstance(series, pd.Series)
    if _series_is_literal_color(series):
        return None
    palette = qualitative_colors.Plotly
    values = [_normalize_option(value) for value in pd.unique(series)]
    return {value: palette[idx % len(palette)] for idx, value in enumerate(values)}


def _build_color_group_specs(
    df: pd.DataFrame,
    color_field: str | None,
    color_cfg: ChannelAestheticsConfig | None,
    color_map: dict[str, str] | None,
) -> list[ColorGroupSpec]:
    if color_field is None or color_field not in df.columns:
        return [ColorGroupSpec(label=None, value=None, constant_color=None, mode="none")]
    series = df[color_field]
    assert isinstance(series, pd.Series)
    if _series_is_literal_color(series):
        return [ColorGroupSpec(label=None, value=None, constant_color=None, mode="literal")]
    if color_cfg and color_cfg.type in {"nominal", "ordinal"}:
        specs: list[ColorGroupSpec] = []
        for value in pd.unique(series):
            normalized = _normalize_option(value)
            constant_color = (color_map or {}).get(normalized)
            specs.append(
                ColorGroupSpec(
                    label=str(value),
                    value=value,
                    constant_color=constant_color,
                    mode="discrete",
                )
            )
        return specs
    if color_cfg:
        return [ColorGroupSpec(label=color_cfg.title or color_field, value=None, constant_color=None, mode="field")]
    return [ColorGroupSpec(label=None, value=None, constant_color=None, mode="none")]


def _normalize_option(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except AttributeError:  # pragma: no cover - defensive
            return value
    return value


def _series_is_literal_color(series: pd.Series) -> bool:
    if series.empty:
        return False
    return series.dropna().map(_value_is_color_string).all()


def _value_is_color_string(value: Any) -> bool:
    if isinstance(value, str):
        candidate = value.strip()
        if _HEX_COLOR_PATTERN.match(candidate):
            return True
        lowered = candidate.lower()
        return lowered.startswith("rgb(") or lowered.startswith("rgba(")
    return False
