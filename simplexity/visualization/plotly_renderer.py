"""Plotly renderer for visualization PlotConfigs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px

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
    """Render a PlotConfig into a Plotly Figure (currently 3D scatter only)."""
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Plotly rendering.")
    if len(plot_cfg.layers) != 1:
        raise ConfigValidationError("Plotly renderer currently supports exactly one layer.")

    layer = plot_cfg.layers[0]
    if layer.geometry.type != "point":
        raise ConfigValidationError("Plotly renderer currently supports point geometry for 3D scatter demo.")

    plot_df = build_plot_level_dataframe(plot_cfg.data, plot_cfg.transforms, data_registry)
    layer_df = resolve_layer_dataframe(layer, plot_df, data_registry)

    figure = _build_scatter3d(layer, layer_df)
    figure = _apply_plot_level_properties(figure, plot_cfg.guides, plot_cfg.size, plot_cfg.background, layer.aesthetics)
    return figure


def _build_scatter3d(layer: LayerConfig, df: pd.DataFrame):
    aes = layer.aesthetics
    x_field = _require_field(aes.x, "x")
    y_field = _require_field(aes.y, "y")
    z_field = _require_field(aes.z, "z")

    color_field = _optional_field(aes.color)
    size_field = _optional_field(aes.size)
    opacity_value = _resolve_opacity(aes.opacity)
    hover_fields = _collect_tooltip_fields(aes.tooltip)

    figure = px.scatter_3d(
        df,
        x=x_field,
        y=y_field,
        z=z_field,
        color=color_field,
        size=size_field,
        hover_data=hover_fields or None,
        opacity=opacity_value,
    )

    if aes.color and aes.color.value is not None:
        figure.update_traces(marker=dict(color=aes.color.value))
    if aes.size and aes.size.value is not None:
        figure.update_traces(marker=dict(size=aes.size.value))

    trace_name = layer.name or (color_field or "3d_scatter")
    figure.update_traces(name=trace_name, selector=dict(type="scatter3d"))
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
