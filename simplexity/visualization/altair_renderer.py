"""Altair renderer for declarative visualization configs."""

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
    AxisConfig,
    ChannelAestheticsConfig,
    FacetConfig,
    GeometryConfig,
    LayerConfig,
    LegendConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
    ScaleConfig,
    SelectionConfig,
)

LOGGER = logging.getLogger(__name__)

_CHANNEL_CLASS_MAP = {
    "x": "X",
    "y": "Y",
    "color": "Color",
    "size": "Size",
    "shape": "Shape",
    "opacity": "Opacity",
    "row": "Row",
    "column": "Column",
}


def build_altair_chart(
    plot_cfg: PlotConfig,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
):
    """Render a PlotConfig into an Altair Chart."""
    alt = _import_altair()
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Altair rendering.")

    plot_df = build_plot_level_dataframe(plot_cfg.data, plot_cfg.transforms, data_registry)

    layer_charts = [
        _build_layer_chart(alt, layer, resolve_layer_dataframe(layer, plot_df, data_registry))
        for layer in plot_cfg.layers
    ]

    chart = layer_charts[0] if len(layer_charts) == 1 else alt.layer(*layer_charts)

    if plot_cfg.selections:
        chart = chart.add_params(*[_build_selection_param(alt, sel) for sel in plot_cfg.selections])

    if plot_cfg.facet:
        chart = _apply_facet(alt, chart, plot_cfg.facet)

    chart = _apply_plot_level_properties(alt, chart, plot_cfg.guides, plot_cfg.size, plot_cfg.background)

    return chart


def _import_altair():
    try:
        import altair as alt  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover - dependency missing only in unsupported envs
        raise ImportError("Altair is required for visualization rendering. Install `altair` to continue.") from exc
    return alt


def _build_layer_chart(alt, layer: LayerConfig, df: pd.DataFrame):
    chart = alt.Chart(df)
    chart = _apply_geometry(chart, layer.geometry)
    encoding_kwargs = _encode_aesthetics(alt, layer.aesthetics)
    if encoding_kwargs:
        chart = chart.encode(**encoding_kwargs)
    if layer.selections:
        chart = chart.add_params(*[_build_selection_param(alt, sel) for sel in layer.selections])
    return chart


def _apply_geometry(chart, geometry: GeometryConfig):
    mark_name = f"mark_{geometry.type}"
    if not hasattr(chart, mark_name):
        raise ConfigValidationError(f"Altair chart does not support geometry type '{geometry.type}'")
    mark_fn = getattr(chart, mark_name)
    return mark_fn(**(geometry.props or {}))


def _encode_aesthetics(alt, aesthetics: AestheticsConfig) -> dict[str, Any]:
    encodings: dict[str, Any] = {}
    for channel_name in ("x", "y", "color", "size", "shape", "opacity", "row", "column"):
        channel_cfg = getattr(aesthetics, channel_name)
        channel_value = _channel_to_alt(alt, channel_name, channel_cfg)
        if channel_value is not None:
            encodings[channel_name] = channel_value

    if aesthetics.tooltip:
        encodings["tooltip"] = [_tooltip_to_alt(alt, tooltip_cfg) for tooltip_cfg in aesthetics.tooltip]

    return encodings


def _channel_to_alt(alt, channel_name: str, cfg: ChannelAestheticsConfig | None):
    if cfg is None:
        return None
    if cfg.value is not None and cfg.field is None:
        return alt.value(cfg.value)
    channel_cls_name = _CHANNEL_CLASS_MAP[channel_name]
    channel_cls = getattr(alt, channel_cls_name)
    kwargs: dict[str, Any] = {}
    if cfg.field:
        kwargs["field"] = cfg.field
    if cfg.type:
        kwargs["type"] = cfg.type
    if cfg.title:
        kwargs["title"] = cfg.title
    if cfg.aggregate:
        kwargs["aggregate"] = cfg.aggregate
    if cfg.bin is not None:
        kwargs["bin"] = cfg.bin
    if cfg.time_unit:
        kwargs["timeUnit"] = cfg.time_unit
    if cfg.sort is not None:
        kwargs["sort"] = alt.Sort(cfg.sort) if isinstance(cfg.sort, list) else cfg.sort
    if cfg.scale:
        kwargs["scale"] = _scale_to_alt(alt, cfg.scale)
    if cfg.axis and channel_name in {"x", "y", "row", "column"}:
        kwargs["axis"] = _axis_to_alt(alt, cfg.axis)
    if cfg.legend and channel_name in {"color", "size", "shape", "opacity"}:
        kwargs["legend"] = _legend_to_alt(alt, cfg.legend)
    return channel_cls(**kwargs)


def _tooltip_to_alt(alt, cfg: ChannelAestheticsConfig):
    if cfg.value is not None and cfg.field is None:
        return alt.Tooltip(value=cfg.value, title=cfg.title)
    if cfg.field is None:
        raise ConfigValidationError("Tooltip channels must set either a field or a constant value.")

    kwargs: dict[str, Any] = {"field": cfg.field}
    if cfg.type:
        kwargs["type"] = cfg.type
    if cfg.title:
        kwargs["title"] = cfg.title
    return alt.Tooltip(**kwargs)


def _scale_to_alt(alt, cfg: ScaleConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Scale(**kwargs)


def _axis_to_alt(alt, cfg: AxisConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Axis(**kwargs)


def _legend_to_alt(alt, cfg: LegendConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Legend(**kwargs)


def _build_selection_param(alt, cfg: SelectionConfig):
    if cfg.type == "interval":
        return alt.selection_interval(name=cfg.name, encodings=cfg.encodings, fields=cfg.fields, bind=cfg.bind)
    if cfg.type == "single":
        return alt.selection_single(name=cfg.name, encodings=cfg.encodings, fields=cfg.fields, bind=cfg.bind)
    if cfg.type == "multi":
        return alt.selection_multi(name=cfg.name, encodings=cfg.encodings, fields=cfg.fields, bind=cfg.bind)
    raise ConfigValidationError(f"Unsupported selection type '{cfg.type}' for Altair renderer.")


def _apply_facet(alt, chart, facet_cfg: FacetConfig):
    facet_args: dict[str, Any] = {}
    if facet_cfg.row:
        facet_args["row"] = alt.Row(facet_cfg.row)
    if facet_cfg.column:
        facet_args["column"] = alt.Column(facet_cfg.column)
    if facet_cfg.wrap:
        raise ConfigValidationError("FacetConfig.wrap is not yet implemented for Altair rendering.")
    if not facet_args:
        return chart
    return chart.facet(**facet_args)


def _apply_plot_level_properties(
    alt, chart, guides: PlotLevelGuideConfig, size: PlotSizeConfig, background: str | None
):
    title_params = _build_title_params(alt, guides)
    if title_params is not None:
        chart = chart.properties(title=title_params)
    width = size.width
    height = size.height
    if width is not None or height is not None:
        chart = chart.properties(width=width, height=height)
    if size.autosize:
        chart.autosize = size.autosize
    if background:
        chart = chart.configure(background=background)
    if guides.labels:
        LOGGER.info("Plot-level labels are not yet implemented for Altair; skipping %s labels.", len(guides.labels))
    return chart


def _build_title_params(alt, guides: PlotLevelGuideConfig):
    subtitle_lines = [text for text in (guides.subtitle, guides.caption) if text]
    if not guides.title and not subtitle_lines:
        return None
    if subtitle_lines:
        return alt.TitleParams(text=guides.title or "", subtitle=subtitle_lines)
    return guides.title
