"""Plotly renderer for visualization PlotConfigs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

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


def build_plotly_figure(
    plot_cfg: PlotConfig,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
):
    """Render a PlotConfig into a Plotly Figure.

    Supports:
    - Multiple layers
    - 2D scatter, 3D scatter, and line geometries
    - Auto-generation of interactive controls (dropdown/slider) from layer naming patterns
    """
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Plotly rendering.")

    plot_df = build_plot_level_dataframe(plot_cfg.data, plot_cfg.transforms, data_registry)

    # Build all layers
    import plotly.graph_objects as go
    figure = go.Figure()

    for layer in plot_cfg.layers:
        layer_df = resolve_layer_dataframe(layer, plot_df, data_registry)

        # Build traces based on geometry type
        if layer.geometry.type == "point":
            if layer.aesthetics.z is not None:
                traces = _build_scatter3d(layer, layer_df)
            else:
                traces = _build_scatter2d(layer, layer_df)
        elif layer.geometry.type == "line":
            traces = _build_line(layer, layer_df)
        else:
            raise ConfigValidationError(
                f"Plotly renderer does not support geometry type '{layer.geometry.type}'. "
                f"Supported types: point (2D/3D), line"
            )

        # Add traces to figure (may be single trace or list)
        if isinstance(traces, list):
            for trace in traces:
                figure.add_trace(trace)
        else:
            figure.add_trace(traces)

    # Apply plot-level properties
    default_aes = plot_cfg.layers[0].aesthetics if plot_cfg.layers else AestheticsConfig()
    figure = _apply_plot_level_properties(
        figure, plot_cfg.guides, plot_cfg.size, plot_cfg.background, default_aes
    )

    # Auto-generate interactive controls from layer naming patterns
    figure = _auto_generate_controls(figure, plot_cfg.layers)

    return figure


def _build_scatter3d(layer: LayerConfig, df: pd.DataFrame):
    """Build 3D scatter trace(s) for a layer."""
    import plotly.graph_objects as go

    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")
    z_field = _require_field(aes.z, "z")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    # Build marker dict
    marker = {}
    if opacity_value is not None:
        marker["opacity"] = opacity_value
    if aes.color and aes.color.value is not None:
        marker["color"] = aes.color.value
    elif color_field:
        # Check if there's a custom scale mapping for nominal/categorical data
        if aes.color and aes.color.scale and aes.color.scale.domain and aes.color.scale.range:
            # Map categorical values to colors using the scale
            color_map = dict(zip(aes.color.scale.domain, aes.color.scale.range))
            marker["color"] = df[color_field].map(color_map)
        else:
            marker["color"] = df[color_field]
    if aes.size and aes.size.value is not None:
        marker["size"] = aes.size.value
    elif size_field:
        marker["size"] = df[size_field]

    # Build hover data
    hovertemplate = None
    if hover_fields:
        hovertemplate = "<br>".join(f"{field}: %{{customdata[{i}]}}" for i, field in enumerate(hover_fields))
        hovertemplate += "<extra></extra>"

    trace = go.Scatter3d(
        x=df[x_field],
        y=df[y_field],
        z=df[z_field],
        mode="markers",
        marker=marker,
        name=layer.name or "3d_scatter",
        customdata=df[hover_fields].values if hover_fields else None,
        hovertemplate=hovertemplate,
        visible=True,  # Will be controlled by auto-generated controls
    )

    return trace


def _build_scatter2d(layer: LayerConfig, df: pd.DataFrame):
    """Build 2D scatter trace(s) for a layer."""
    import plotly.graph_objects as go

    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    # Build marker dict
    marker = {}
    if opacity_value is not None:
        marker["opacity"] = opacity_value
    if aes.color and aes.color.value is not None:
        marker["color"] = aes.color.value
    elif color_field:
        # Check if there's a custom scale mapping for nominal/categorical data
        if aes.color and aes.color.scale and aes.color.scale.domain and aes.color.scale.range:
            # Map categorical values to colors using the scale
            color_map = dict(zip(aes.color.scale.domain, aes.color.scale.range))
            marker["color"] = df[color_field].map(color_map)
        else:
            marker["color"] = df[color_field]
            # Add colorscale if it's a numeric field
            if pd.api.types.is_numeric_dtype(df[color_field]):
                marker["colorscale"] = "Viridis"
                marker["showscale"] = True
    if aes.size and aes.size.value is not None:
        marker["size"] = aes.size.value
    elif size_field:
        marker["size"] = df[size_field]

    # Build hover data
    hovertemplate = None
    if hover_fields:
        hovertemplate = "<br>".join(f"{field}: %{{customdata[{i}]}}" for i, field in enumerate(hover_fields))
        hovertemplate += "<extra></extra>"

    trace = go.Scatter(
        x=df[x_field],
        y=df[y_field],
        mode="markers",
        marker=marker,
        name=layer.name or "2d_scatter",
        customdata=df[hover_fields].values if hover_fields else None,
        hovertemplate=hovertemplate,
        visible=True,  # Will be controlled by auto-generated controls
    )

    return trace


def _build_line(layer: LayerConfig, df: pd.DataFrame):
    """Build line trace(s) for a layer."""
    import plotly.graph_objects as go

    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")

    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    # Build line dict
    line = {}
    if opacity_value is not None:
        line["opacity"] = opacity_value if opacity_value else 1.0
    if aes.color and aes.color.value is not None:
        line["color"] = aes.color.value
    # Apply any geometry props (e.g., width, dash)
    line.update(layer.geometry.props)

    # Build hover data
    hovertemplate = None
    if hover_fields:
        hovertemplate = "<br>".join(f"{field}: %{{customdata[{i}]}}" for i, field in enumerate(hover_fields))
        hovertemplate += "<extra></extra>"

    trace = go.Scatter(
        x=df[x_field],
        y=df[y_field],
        mode="lines+markers" if layer.geometry.props.get("show_markers", False) else "lines",
        line=line if line else None,
        name=layer.name or "line",
        customdata=df[hover_fields].values if hover_fields else None,
        hovertemplate=hovertemplate,
        visible=True,  # Will be controlled by auto-generated controls
    )

    return trace


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

    scene_updates: dict[str, Any] = {}
    x_title = _axis_title(aes.x)
    y_title = _axis_title(aes.y)
    z_title = _axis_title(aes.z)
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


def _auto_generate_controls(figure, layers: list[LayerConfig]):
    """Auto-generate interactive controls (dropdown/slider) from layer naming patterns.

    Detects naming patterns like:
    - "step_{n}__layer_{name}" → generates dropdown for layers, slider for steps
    - "layer_{name}__step_{n}" → generates dropdown for layers, slider for steps
    - "step_{n}" → generates slider for steps only
    - "layer_{name}" → generates dropdown for layers only

    Args:
        figure: Plotly Figure object
        layers: List of layer configurations

    Returns:
        Updated figure with controls
    """
    if not layers:
        return figure

    # Parse layer names to extract step and layer information
    step_layer_map = {}  # {(step, layer): [trace_indices]}
    steps = set()
    layer_names = set()
    has_step_layer_pattern = False

    for layer_idx, layer in enumerate(layers):
        if not layer.name:
            continue

        # Try to parse "step_{n}__layer_{name}" or "layer_{name}__step_{n}"
        if "__" in layer.name:
            parts = layer.name.split("__")
            step_val = None
            layer_val = None

            for part in parts:
                if part.startswith("step_"):
                    try:
                        step_val = int(part.replace("step_", ""))
                        steps.add(step_val)
                    except ValueError:
                        pass
                elif part.startswith("layer_"):
                    layer_val = part.replace("layer_", "")
                    layer_names.add(layer_val)

            if step_val is not None and layer_val is not None:
                has_step_layer_pattern = True
                key = (step_val, layer_val)
                if key not in step_layer_map:
                    step_layer_map[key] = []
                step_layer_map[key].append(layer_idx)

    # If we found the pattern, generate controls
    if has_step_layer_pattern and steps and layer_names:
        steps_sorted = sorted(steps)
        layers_sorted = sorted(layer_names)

        # Build slider for steps (controls which step is visible within selected layer)
        sliders_by_layer = {}
        for layer_name in layers_sorted:
            slider_steps = []
            for _, step in enumerate(steps_sorted):
                # Determine visibility for this step + layer combination
                visible = []
                for trace_idx in range(len(figure.data)):
                    # Check if this trace belongs to (step, layer_name)
                    key = (step, layer_name)
                    visible.append(trace_idx in step_layer_map.get(key, []))

                title_text = figure.layout.title.text if figure.layout.title else ""
                slider_steps.append({
                    "method": "update",
                    "args": [
                        {"visible": visible},
                        {"title": f"{title_text} (Step {step}, {layer_name})"},
                    ],
                    "label": str(step),
                })

            sliders_by_layer[layer_name] = {
                "active": 0,
                "yanchor": "top",
                "y": -0.1,
                "xanchor": "left",
                "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.05,
                "steps": slider_steps,
            }

        # Build dropdown for layers (switches between layers and their corresponding sliders)
        buttons = []
        for _, layer_name in enumerate(layers_sorted):
            # Show first step of this layer
            visible = []
            for trace_idx in range(len(figure.data)):
                key = (steps_sorted[0], layer_name)
                visible.append(trace_idx in step_layer_map.get(key, []))

            title_text = figure.layout.title.text if figure.layout.title else ""
            button = {
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "sliders": [sliders_by_layer[layer_name]],
                        "title": f"{title_text} (Step {steps_sorted[0]}, {layer_name})",
                    },
                ],
                "label": layer_name,
            }
            buttons.append(button)

        # Apply controls to figure
        figure.update_layout(
            sliders=[sliders_by_layer[layers_sorted[0]]],  # Start with first layer's slider
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }]
        )

        # Set initial visibility (first step, first layer)
        for trace_idx in range(len(figure.data)):
            key = (steps_sorted[0], layers_sorted[0])
            figure.data[trace_idx].visible = trace_idx in step_layer_map.get(key, [])

    return figure
