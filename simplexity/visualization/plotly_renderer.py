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
from plotly.subplots import make_subplots

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_pipeline import (
    build_plot_level_dataframe,
    resolve_layer_dataframe,
)
from simplexity.visualization.data_registry import DataRegistry
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    FacetConfig,
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

    # Handle faceting
    if plot_cfg.facet:
        figure = _build_faceted_figure(layer, layer_df, plot_cfg.facet, plot_cfg.size, controls)
        # Use empty size config to avoid overwriting the computed facet dimensions
        empty_size = PlotSizeConfig(width=None, height=None)
        figure = _apply_plot_level_properties(
            figure, plot_cfg.guides, empty_size, plot_cfg.background, layer.aesthetics
        )
        return figure

    has_z = bool(layer.aesthetics and layer.aesthetics.z and layer.aesthetics.z.field)
    if has_z:
        figure = _build_scatter3d(layer, layer_df, controls)
    else:
        figure = _build_scatter2d(layer, layer_df, controls)
    figure = _apply_plot_level_properties(figure, plot_cfg.guides, plot_cfg.size, plot_cfg.background, layer.aesthetics)
    return figure


def _build_faceted_figure(
    layer: LayerConfig,
    df: pd.DataFrame,
    facet_cfg: FacetConfig,
    size_cfg: PlotSizeConfig,
    controls: Any | None,
):
    """Build a faceted subplot figure."""
    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")
    has_z = bool(aes.z and aes.z.field)
    z_field = _require_field(aes.z, "z") if has_z else None

    row_field = facet_cfg.row
    col_field = facet_cfg.column

    # Resolve controls
    dropdown = _resolve_layer_dropdown(df, controls)
    slider_enabled = not (getattr(controls, "accumulate_steps", False))
    slider = _resolve_slider_control(df, controls if slider_enabled else None)

    # Filter by initial layer if dropdown is active
    # Keep rows that don't depend on layer (e.g., ground truth from belief states with layer="_no_layer_")
    layer_field, layer_options = dropdown if dropdown else (None, [None])
    working_df: pd.DataFrame
    if layer_field is None:
        working_df = df
    else:
        layer_independent_filter = df[df[layer_field] == "_no_layer_"]
        layer_dependent_filter = df[(df[layer_field] != "_no_layer_") & (df[layer_field] == layer_options[0])]
        assert isinstance(layer_independent_filter, pd.DataFrame)
        assert isinstance(layer_dependent_filter, pd.DataFrame)
        working_df = pd.concat([layer_dependent_filter, layer_independent_filter], ignore_index=True)

    # Get unique values for faceting dimensions
    # Use dict.fromkeys to deduplicate by string representation (handles mixed int/str types)
    row_values: list[str | None]
    col_values: list[str | None]
    if row_field:
        raw_row = list(pd.unique(working_df[row_field]))
        row_values = list(dict.fromkeys(str(v) for v in sorted(raw_row, key=str)))
    else:
        row_values = [None]
    if col_field:
        raw_col = list(pd.unique(working_df[col_field]))
        col_values = list(dict.fromkeys(str(v) for v in sorted(raw_col, key=str)))
    else:
        col_values = [None]

    n_rows = len(row_values)
    n_cols = len(col_values)

    # Build subplot titles
    subplot_titles = []
    for row_val in row_values:
        for col_val in col_values:
            if row_val is not None and col_val is not None:
                subplot_titles.append(f"{col_val}")
            elif col_val is not None:
                subplot_titles.append(str(col_val))
            elif row_val is not None:
                subplot_titles.append(str(row_val))
            else:
                subplot_titles.append("")

    # Create subplot grid with appropriate specs for 2D or 3D
    if has_z:
        specs = [[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
            specs=specs,
            row_titles=[str(v) for v in row_values] if row_field else None,
        )
    else:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            row_titles=[str(v) for v in row_values] if row_field else None,
        )

    color_field = _optional_field(aes.color)
    size_field_name = _optional_field(aes.size)
    size_value = _resolve_size_value(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)
    color_map = _build_color_discrete_map(df, color_field, aes.color)
    color_specs = _build_color_group_specs(df, color_field, aes.color, color_map)

    # Helper to build traces for a given filtered dataframe
    def build_facet_traces(source_df: pd.DataFrame, show_legend: bool = True):
        traces_by_cell: dict[tuple[int, int], list[Any]] = {}
        for row_idx, row_val in enumerate(row_values, start=1):
            for col_idx, col_val in enumerate(col_values, start=1):
                cell_df = source_df.copy()
                if row_field:
                    filtered = cell_df[cell_df[row_field].astype(str) == row_val]
                    assert isinstance(filtered, pd.DataFrame)
                    cell_df = filtered
                if col_field:
                    filtered = cell_df[cell_df[col_field].astype(str) == col_val]
                    assert isinstance(filtered, pd.DataFrame)
                    cell_df = filtered

                if cell_df.empty:
                    traces_by_cell[(row_idx, col_idx)] = []
                    continue

                if has_z:
                    assert z_field is not None
                    traces = _scatter3d_traces(
                        cell_df,
                        x_field,
                        y_field,
                        z_field,
                        color_field,
                        size_field_name,
                        hover_fields,
                        opacity_value,
                        color_specs,
                        layer_name=layer.name,
                        size_value=size_value,
                    )
                    scene_idx = (row_idx - 1) * n_cols + col_idx
                    scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                    for trace in traces:
                        trace.scene = scene_name
                else:
                    traces = _scatter2d_traces(
                        cell_df,
                        x_field,
                        y_field,
                        color_field,
                        size_field_name,
                        hover_fields,
                        opacity_value,
                        color_specs,
                        layer_name=layer.name,
                        size_value=size_value,
                    )

                # Control legend visibility
                for trace in traces:
                    if not show_legend or row_idx > 1 or col_idx > 1:
                        trace.showlegend = False

                traces_by_cell[(row_idx, col_idx)] = traces
        return traces_by_cell

    # Get slider values if slider is active
    slider_field, slider_values = slider if slider else (None, [None])

    if slider and layer_field:
        assert slider_field is not None
        # Both slider and dropdown: complex case with frames per (layer, step)
        # For simplicity, build frames for current layer only
        initial_step = slider_values[0] if slider_values else None
        initial_df = working_df
        if initial_step is not None and slider_field in working_df.columns:
            filtered = working_df[working_df[slider_field] == initial_step]
            assert isinstance(filtered, pd.DataFrame)
            initial_df = filtered

        traces_by_cell = build_facet_traces(initial_df)
        for (row_idx, col_idx), traces in traces_by_cell.items():
            for trace in traces:
                fig.add_trace(trace, row=row_idx, col=col_idx)

        # Build frames for slider animation
        frames = []
        for step_val in slider_values:
            step_filtered = working_df[working_df[slider_field] == step_val]
            assert isinstance(step_filtered, pd.DataFrame)
            frame_traces_by_cell = build_facet_traces(step_filtered, show_legend=False)
            frame_traces: list[Any] = []
            for row_idx, col_idx in sorted(frame_traces_by_cell.keys()):
                frame_traces.extend(frame_traces_by_cell[(row_idx, col_idx)])
            frames.append(go.Frame(name=str(step_val), data=frame_traces))
        fig.frames = frames
        _add_slider_layout(fig, slider_field, slider_values)

        # Add layer dropdown
        if len(layer_options) > 1:
            _add_faceted_layer_dropdown(
                fig,
                df,
                layer_field,
                layer_options,
                slider_field,
                slider_values,
                row_values,
                col_values,
                row_field,
                col_field,
                x_field,
                y_field,
                z_field,
                color_field,
                size_field_name,
                hover_fields,
                opacity_value,
                color_specs,
                layer,
                has_z,
                n_cols,
                size_value,
            )

    elif slider:
        assert slider_field is not None
        # Slider only
        initial_step = slider_values[0] if slider_values else None
        initial_df = working_df
        if initial_step is not None and slider_field in working_df.columns:
            filtered = working_df[working_df[slider_field] == initial_step]
            assert isinstance(filtered, pd.DataFrame)
            initial_df = filtered

        traces_by_cell = build_facet_traces(initial_df)
        for (row_idx, col_idx), traces in traces_by_cell.items():
            for trace in traces:
                fig.add_trace(trace, row=row_idx, col=col_idx)

        # Build frames for slider animation
        frames = []
        for step_val in slider_values:
            step_filtered = working_df[working_df[slider_field] == step_val]
            assert isinstance(step_filtered, pd.DataFrame)
            frame_traces_by_cell = build_facet_traces(step_filtered, show_legend=False)
            frame_traces: list[Any] = []
            for row_idx, col_idx in sorted(frame_traces_by_cell.keys()):
                frame_traces.extend(frame_traces_by_cell[(row_idx, col_idx)])
            frames.append(go.Frame(name=str(step_val), data=frame_traces))
        fig.frames = frames
        _add_slider_layout(fig, slider_field, slider_values)

    elif dropdown and len(layer_options) > 1:
        assert layer_field is not None
        # Dropdown only
        traces_by_cell = build_facet_traces(working_df)
        for (row_idx, col_idx), traces in traces_by_cell.items():
            for trace in traces:
                fig.add_trace(trace, row=row_idx, col=col_idx)

        _add_faceted_layer_dropdown(
            fig,
            df,
            layer_field,
            layer_options,
            None,
            [],
            row_values,
            col_values,
            row_field,
            col_field,
            x_field,
            y_field,
            z_field,
            color_field,
            size_field_name,
            hover_fields,
            opacity_value,
            color_specs,
            layer,
            has_z,
            n_cols,
            size_value,
        )
    else:
        # No controls
        traces_by_cell = build_facet_traces(working_df)
        for (row_idx, col_idx), traces in traces_by_cell.items():
            for trace in traces:
                fig.add_trace(trace, row=row_idx, col=col_idx)

    # Apply size to individual subplots if specified
    subplot_width = size_cfg.width or 200
    subplot_height = size_cfg.height or 200
    total_width = subplot_width * n_cols + 100  # Extra space for margins
    total_height = subplot_height * n_rows + 100

    fig.update_layout(
        width=total_width,
        height=total_height,
        showlegend=True,
    )

    # For 3D, set axis titles, ranges, and aspect ratio on each scene
    if has_z:
        x_title = _axis_title(aes.x)
        y_title = _axis_title(aes.y)
        z_title = _axis_title(aes.z)
        x_range = _axis_domain(aes.x)
        y_range = _axis_domain(aes.y)
        z_range = _axis_domain(aes.z)
        for row_idx in range(1, n_rows + 1):
            for col_idx in range(1, n_cols + 1):
                scene_idx = (row_idx - 1) * n_cols + col_idx
                scene_key = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                scene_update: dict[str, Any] = {"aspectmode": "cube"}
                xaxis_cfg: dict[str, Any] = {}
                yaxis_cfg: dict[str, Any] = {}
                zaxis_cfg: dict[str, Any] = {}
                if x_title:
                    xaxis_cfg["title"] = x_title
                if y_title:
                    yaxis_cfg["title"] = y_title
                if z_title:
                    zaxis_cfg["title"] = z_title
                if x_range:
                    xaxis_cfg["range"] = x_range
                if y_range:
                    yaxis_cfg["range"] = y_range
                if z_range:
                    zaxis_cfg["range"] = z_range
                if xaxis_cfg:
                    scene_update["xaxis"] = xaxis_cfg
                if yaxis_cfg:
                    scene_update["yaxis"] = yaxis_cfg
                if zaxis_cfg:
                    scene_update["zaxis"] = zaxis_cfg
                layout_update: dict[str, Any] = {scene_key: scene_update}
                fig.update_layout(**layout_update)

    return fig


def _add_faceted_layer_dropdown(
    fig: go.Figure,
    df: pd.DataFrame,
    layer_field: str,
    layer_options: list[Any],
    slider_field: str | None,
    slider_values: list[Any],
    row_values: list[str | None],
    col_values: list[str | None],
    row_field: str | None,
    col_field: str | None,
    x_field: str,
    y_field: str,
    z_field: str | None,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    layer: LayerConfig,
    has_z: bool,
    n_cols: int,
    size_value: float | None = None,
) -> None:
    """Add a layer dropdown menu that rebuilds traces for faceted figures."""
    # Get layer-independent rows (e.g., ground truth from belief states)
    layer_independent_filtered = df[df[layer_field] == "_no_layer_"]
    assert isinstance(layer_independent_filtered, pd.DataFrame)
    layer_independent = layer_independent_filtered

    buttons = []
    for layer_opt in layer_options:
        # Combine layer-specific rows with layer-independent rows
        layer_specific_filtered = df[(df[layer_field] != "_no_layer_") & (df[layer_field] == layer_opt)]
        assert isinstance(layer_specific_filtered, pd.DataFrame)
        layer_df = pd.concat([layer_specific_filtered, layer_independent], ignore_index=True)

        # If there's a slider, filter to initial step
        if slider_field and slider_values:
            filtered = layer_df[layer_df[slider_field] == slider_values[0]]
            assert isinstance(filtered, pd.DataFrame)
            layer_df = filtered

        # Build traces for this layer
        all_traces: list[Any] = []
        for row_idx, row_val in enumerate(row_values, start=1):
            for col_idx, col_val in enumerate(col_values, start=1):
                cell_df = layer_df.copy()
                if row_field:
                    filtered = cell_df[cell_df[row_field].astype(str) == row_val]
                    assert isinstance(filtered, pd.DataFrame)
                    cell_df = filtered
                if col_field:
                    filtered = cell_df[cell_df[col_field].astype(str) == col_val]
                    assert isinstance(filtered, pd.DataFrame)
                    cell_df = filtered

                if cell_df.empty:
                    continue

                if has_z:
                    assert z_field is not None
                    traces = _scatter3d_traces(
                        cell_df,
                        x_field,
                        y_field,
                        z_field,
                        color_field,
                        size_field,
                        hover_fields,
                        opacity_value,
                        color_specs,
                        layer_name=layer.name,
                        size_value=size_value,
                    )
                    scene_idx = (row_idx - 1) * n_cols + col_idx
                    scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                    for trace in traces:
                        trace.scene = scene_name
                        trace.showlegend = False
                else:
                    traces = _scatter2d_traces(
                        cell_df,
                        x_field,
                        y_field,
                        color_field,
                        size_field,
                        hover_fields,
                        opacity_value,
                        color_specs,
                        layer_name=layer.name,
                        size_value=size_value,
                    )
                    for trace in traces:
                        trace.showlegend = False

                all_traces.extend(traces)

        # Create button that replaces all trace data
        button = {
            "label": str(layer_opt),
            "method": "restyle",
            "args": [
                {
                    "x": [list(t.x) if hasattr(t, "x") else [] for t in all_traces],
                    "y": [list(t.y) if hasattr(t, "y") else [] for t in all_traces],
                }
            ],
        }
        if has_z:
            button["args"][0]["z"] = [list(t.z) if hasattr(t, "z") else [] for t in all_traces]

        buttons.append(button)

    fig.update_layout(
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


def _build_scatter3d(layer: LayerConfig, df: pd.DataFrame, controls: Any | None):
    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")
    z_field = _require_field(aes.z, "z")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    size_value = _resolve_size_value(aes.size)
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
            size_value,
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
            size_value,
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
            size_value=size_value,
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
    size_value = _resolve_size_value(aes.size)
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
            size_value,
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
            size_value,
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
            size_value=size_value,
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


def _resolve_size_value(channel: ChannelAestheticsConfig | None) -> float | None:
    if channel is None or channel.value is None:
        return None
    try:
        return float(channel.value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError("Size channel value must be numeric.") from exc


def _axis_title(channel: ChannelAestheticsConfig | None) -> str | None:
    if channel is None:
        return None
    return channel.title or channel.field


def _axis_domain(channel: ChannelAestheticsConfig | None) -> list[Any] | None:
    if channel is None or channel.scale is None:
        return None
    return channel.scale.domain


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
    # Filter out "_no_layer_" placeholder used for layer-independent data
    valid_values = [value for value in options if value in set(df[field_name]) and value != "_no_layer_"]
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
    size_value: float | None = None,
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
            size_value=size_value,
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
            size_value=size_value,
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
    size_value: float | None = None,
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
            size_value=size_value,
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
            size_value=size_value,
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


def _build_slider_scatter(
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
    *,
    z_field: str | None = None,
    size_value: float | None = None,
):
    """Build slider-controlled scatter plot (2D or 3D based on z_field presence)."""
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
        subset_traces = _scatter_traces(
            initial_subset,
            x_field,
            y_field,
            color_field,
            size_field,
            hover_fields,
            opacity_value,
            color_specs,
            z_field=z_field,
            layer_name=layer_label,
            keep_empty=True,
            size_value=size_value,
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
            frame_traces = _scatter_traces(
                slider_subset,
                x_field,
                y_field,
                color_field,
                size_field,
                hover_fields,
                opacity_value,
                color_specs,
                z_field=z_field,
                layer_name=layer_label,
                keep_empty=True,
                size_value=size_value,
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
    size_value: float | None = None,
):
    """Build 3D slider scatter. Wrapper around _build_slider_scatter."""
    return _build_slider_scatter(
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
        z_field=z_field,
        size_value=size_value,
    )


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
    size_value: float | None = None,
):
    """Build 2D slider scatter. Wrapper around _build_slider_scatter."""
    return _build_slider_scatter(
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
        z_field=None,
        size_value=size_value,
    )


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


def _scatter_traces(
    df: pd.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str | None,
    size_field: str | None,
    hover_fields: list[str],
    opacity_value: float | None,
    color_specs: list[ColorGroupSpec],
    *,
    z_field: str | None = None,
    layer_name: str | None = None,
    keep_empty: bool = False,
    size_value: float | None = None,
) -> list[go.Scatter3d] | list[go.Scatter]:
    """Build scatter traces (2D or 3D based on z_field presence)."""
    is_3d = z_field is not None
    traces: list[Any] = []

    for idx, spec in enumerate(color_specs):
        subset = _subset_for_spec(df, color_field, spec)
        if subset.empty and not keep_empty:
            continue
        marker = _build_marker(subset, color_field, size_field, spec, size_value)
        customdata = _build_customdata(subset, hover_fields)

        if is_3d:
            assert z_field is not None
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
        else:
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
        default_name = layer_name or ("scatter3d" if is_3d else "scatter")
        if is_3d:
            empty_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", name=default_name)
        else:
            empty_trace = go.Scatter(x=[], y=[], mode="markers", name=default_name)
        if opacity_value is not None:
            empty_trace.opacity = opacity_value
        traces.append(empty_trace)

    return traces


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
    size_value: float | None = None,
) -> list[go.Scatter3d]:
    """Build 3D scatter traces. Wrapper around _scatter_traces for type safety."""
    return _scatter_traces(
        df,
        x_field,
        y_field,
        color_field,
        size_field,
        hover_fields,
        opacity_value,
        color_specs,
        z_field=z_field,
        layer_name=layer_name,
        keep_empty=keep_empty,
        size_value=size_value,
    )  # type: ignore[return-value]


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
    size_value: float | None = None,
) -> list[go.Scatter]:
    """Build 2D scatter traces. Wrapper around _scatter_traces for type safety."""
    return _scatter_traces(
        df,
        x_field,
        y_field,
        color_field,
        size_field,
        hover_fields,
        opacity_value,
        color_specs,
        z_field=None,
        layer_name=layer_name,
        keep_empty=keep_empty,
        size_value=size_value,
    )  # type: ignore[return-value]


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
    size_value: float | None = None,
) -> dict[str, Any]:
    marker: dict[str, Any] = {}
    if size_field and size_field in df.columns:
        marker["size"] = df[size_field].tolist()
    elif size_value is not None:
        marker["size"] = size_value
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
