"""Altair renderer for declarative visualization configs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

try:
    import altair as alt  # type: ignore import-not-found
except ImportError as exc:  # pragma: no cover - dependency missing only in unsupported envs
    raise ImportError("Altair is required for visualization rendering. Install `altair` to continue.") from exc

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
    "detail": "Detail",
}


def build_altair_chart(
    plot_cfg: PlotConfig,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
    controls: Any | None = None,
):
    """Render a PlotConfig into an Altair Chart."""
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Altair rendering.")

    plot_df = build_plot_level_dataframe(plot_cfg.data, plot_cfg.transforms, data_registry)

    layer_charts = [
        _build_layer_chart(layer, resolve_layer_dataframe(layer, plot_df, data_registry)) for layer in plot_cfg.layers
    ]
    layer_charts = _apply_accumulation_detail(layer_charts, plot_cfg.layers, plot_cfg, plot_df, controls)

    chart = layer_charts[0] if len(layer_charts) == 1 else alt.layer(*layer_charts)

    if plot_cfg.selections:
        chart = chart.add_params(*[_build_selection_param(sel) for sel in plot_cfg.selections])

    if plot_cfg.facet:
        chart = _apply_facet(chart, plot_cfg.facet)

    chart = _apply_plot_level_properties(chart, plot_cfg.guides, plot_cfg.size, plot_cfg.background)
    chart = _apply_chart_controls(chart, controls)
    chart = _apply_default_legend_interactivity(chart, plot_cfg.layers)

    return chart


def _build_layer_chart(layer: LayerConfig, df: pd.DataFrame):
    chart = alt.Chart(df)
    chart = _apply_geometry(chart, layer.geometry)
    encoding_kwargs = _encode_aesthetics(layer.aesthetics)
    if encoding_kwargs:
        chart = chart.encode(**encoding_kwargs)
    if layer.selections:
        chart = chart.add_params(*[_build_selection_param(sel) for sel in layer.selections])
    return chart


def _apply_geometry(chart, geometry: GeometryConfig):
    mark_name = f"mark_{geometry.type}"
    if not hasattr(chart, mark_name):
        raise ConfigValidationError(f"Altair chart does not support geometry type '{geometry.type}'")
    mark_fn = getattr(chart, mark_name)
    return mark_fn(**(geometry.props or {}))


def _encode_aesthetics(aesthetics: AestheticsConfig) -> dict[str, Any]:
    encodings: dict[str, Any] = {}
    for channel_name in ("x", "y", "color", "size", "shape", "opacity", "row", "column", "detail"):
        channel_cfg = getattr(aesthetics, channel_name)
        channel_value = _channel_to_alt(channel_name, channel_cfg)
        if channel_value is not None:
            encodings[channel_name] = channel_value

    if aesthetics.tooltip:
        encodings["tooltip"] = [_tooltip_to_alt(tooltip_cfg) for tooltip_cfg in aesthetics.tooltip]

    return encodings


def _channel_to_alt(channel_name: str, cfg: ChannelAestheticsConfig | None):
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
        kwargs["scale"] = _scale_to_alt(cfg.scale)
    if cfg.axis and channel_name in {"x", "y", "row", "column"}:
        kwargs["axis"] = _axis_to_alt(cfg.axis)
    if cfg.legend and channel_name in {"color", "size", "shape", "opacity"}:
        if cfg.legend.visible is False:
            kwargs["legend"] = None
        else:
            kwargs["legend"] = _legend_to_alt(cfg.legend)
    return channel_cls(**kwargs)


def _tooltip_to_alt(cfg: ChannelAestheticsConfig):
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


def _scale_to_alt(cfg: ScaleConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Scale(**kwargs)


def _axis_to_alt(cfg: AxisConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Axis(**kwargs)


def _legend_to_alt(cfg: LegendConfig):
    kwargs = {k: v for k, v in vars(cfg).items() if v is not None}
    return alt.Legend(**kwargs)


def _build_selection_param(cfg: SelectionConfig):
    kwargs: dict[str, Any] = {}
    if cfg.name:
        kwargs["name"] = cfg.name
    if cfg.encodings:
        kwargs["encodings"] = cfg.encodings
    if cfg.fields:
        kwargs["fields"] = cfg.fields
    if cfg.bind:
        kwargs["bind"] = cfg.bind
    if cfg.type == "interval":
        return alt.selection_interval(**kwargs)
    if cfg.type == "single":
        return alt.selection_single(**kwargs)
    if cfg.type == "multi":
        return alt.selection_multi(**kwargs)
    raise ConfigValidationError(f"Unsupported selection type '{cfg.type}' for Altair renderer.")


def _apply_facet(chart, facet_cfg: FacetConfig):
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


def _apply_plot_level_properties(chart, guides: PlotLevelGuideConfig, size: PlotSizeConfig, background: str | None):
    title_params = _build_title_params(guides)
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


def _apply_chart_controls(chart, controls: Any | None):
    if not controls:
        return chart
    chart = _apply_dropdown_control(chart, getattr(controls, "dropdown", None))
    slider_detail = None if getattr(controls, "accumulate_steps", False) else getattr(controls, "slider", None)
    chart = _apply_slider_control(chart, slider_detail)
    return chart


def _apply_dropdown_control(chart, dropdown):
    field_name = getattr(dropdown, "field", None)
    if dropdown and field_name == "layer":
        options = [_normalize_control_value(value) for value in getattr(dropdown, "options", []) or []]
        if len(options) > 1:
            binding = alt.binding_select(options=options, name="Layer: ")
            param = alt.param(name=f"{field_name}_dropdown", bind=binding, value=options[0])
            return chart.add_params(param).transform_filter(f"datum.{field_name} == {param.name}")
    return chart


def _apply_slider_control(chart, slider):
    field_name = getattr(slider, "field", None)
    options = [_normalize_control_value(value) for value in getattr(slider, "options", []) or []]
    if not slider or not field_name or len(options) <= 1:
        return chart

    numeric_options = _numeric_control_values(options)
    if numeric_options:
        min_val, max_val = numeric_options[0], numeric_options[-1]
        step = _infer_slider_step(numeric_options)
        binding = alt.binding_range(min=min_val, max=max_val, step=step, name=f"{field_name}: ")
        initial_value = numeric_options[0]
    else:
        binding = alt.binding_select(options=options, name=f"{field_name}: ")
        initial_value = options[0]

    param = alt.param(name=f"{field_name}_slider", bind=binding, value=initial_value)
    return chart.add_params(param).transform_filter(f"datum.{field_name} == {param.name}")


def _apply_default_legend_interactivity(chart, layers: list[LayerConfig]):
    if not layers:
        return chart
    color_fields: set[str] = set()
    for layer in layers:
        aesthetics = layer.aesthetics
        if aesthetics and aesthetics.color and aesthetics.color.field:
            color_fields.add(aesthetics.color.field)
    if len(color_fields) != 1:
        return chart
    if any(layer.aesthetics and layer.aesthetics.opacity is not None for layer in layers):
        return chart
    field_name = next(iter(color_fields))
    legend_selection = alt.selection_multi(fields=[field_name], bind="legend", toggle=True, empty="all")
    chart = chart.add_params(legend_selection)
    opacity_encoding = alt.condition(legend_selection, alt.value(1.0), alt.value(0.05))
    return chart.encode(opacity=opacity_encoding)


def _normalize_control_value(value):
    return value.item() if hasattr(value, "item") else value


def _numeric_control_values(options: list[Any]) -> list[float]:
    numeric: list[float] = []
    for value in options:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return []
    numeric = sorted(dict.fromkeys(numeric))
    return numeric


def _infer_slider_step(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = [round(values[idx + 1] - values[idx], 10) for idx in range(len(values) - 1)]
    # Use smallest positive difference or default to 1.0
    step = min((diff for diff in diffs if diff > 0), default=1.0)
    return step


def _apply_accumulation_detail(layer_charts, layers, plot_cfg, plot_df: pd.DataFrame, controls: Any | None):
    if not controls or not getattr(controls, "accumulate_steps", False):
        return layer_charts
    if "step" not in plot_df.columns:
        return layer_charts
    updated = []
    for chart, layer_cfg in zip(layer_charts, layers):
        aesthetics = layer_cfg.aesthetics
        if aesthetics and aesthetics.detail is not None:
            updated.append(chart)
            continue
        if _layer_references_field(layer_cfg, "step"):
            updated.append(chart)
            continue
        updated.append(chart.encode(detail=alt.Detail(field="step", type="ordinal")))
    return updated


def _layer_references_field(layer_cfg: LayerConfig, field: str) -> bool:
    aesthetics = layer_cfg.aesthetics
    if not aesthetics:
        return False

    channel_names = [
        "x",
        "y",
        "x2",
        "y2",
        "color",
        "stroke",
        "strokeDash",
        "size",
        "shape",
        "tooltip",
    ]
    for name in channel_names:
        channel = getattr(aesthetics, name, None)
        if channel is None:
            continue
        # tooltip can be list-like
        if isinstance(channel, list):
            for entry in channel:
                if getattr(entry, "field", None) == field:
                    return True
            continue
        if getattr(channel, "field", None) == field:
            return True
    return False


def _build_title_params(guides: PlotLevelGuideConfig):
    subtitle_lines = [text for text in (guides.subtitle, guides.caption) if text]
    if not guides.title and not subtitle_lines:
        return None
    if subtitle_lines:
        return alt.TitleParams(text=guides.title or "", subtitle=subtitle_lines)
    return guides.title
