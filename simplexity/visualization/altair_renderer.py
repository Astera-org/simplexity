"""Altair renderer for declarative visualization configs."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_registry import DataRegistry, resolve_data_source
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    AxisConfig,
    ChannelAestheticsConfig,
    DataConfig,
    FacetConfig,
    GeometryConfig,
    LayerConfig,
    LegendConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
    ScaleConfig,
    SelectionConfig,
    TransformConfig,
)

LOGGER = logging.getLogger(__name__)

_CALC_ENV = {
    "np": np,
    "pd": pd,
    "math": math,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "clip": np.clip,
}

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


def build_altair_chart(plot_cfg: PlotConfig, data_registry: DataRegistry | Mapping[str, pd.DataFrame]):
    """Render a PlotConfig into an Altair chart."""
    if not plot_cfg.layers:
        raise ConfigValidationError("PlotConfig.layers must include at least one layer for Altair rendering")

    plot_df = _materialize_data(plot_cfg.data, data_registry)
    plot_df = _apply_transforms(plot_df, plot_cfg.transforms)

    layer_charts = [
        _build_layer_chart(layer, _resolve_layer_dataframe(layer, plot_df, data_registry)) for layer in plot_cfg.layers
    ]

    chart = layer_charts[0] if len(layer_charts) == 1 else alt.layer(*layer_charts)

    if plot_cfg.selections:
        chart = chart.add_params(*[_build_selection_param(sel) for sel in plot_cfg.selections])

    if plot_cfg.facet:
        chart = _apply_facet(chart, plot_cfg.facet)

    chart = _apply_plot_level_properties(chart, plot_cfg.guides, plot_cfg.size, plot_cfg.background)

    return chart


def _materialize_data(data_cfg: DataConfig, data_registry: DataRegistry | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    df = resolve_data_source(data_cfg.source, data_registry).copy()
    if data_cfg.filters:
        df = _apply_filters(df, data_cfg.filters)
    if data_cfg.columns:
        missing = [col for col in data_cfg.columns if col not in df.columns]
        if missing:
            raise ConfigValidationError(f"Columns {missing} are not present in data source '{data_cfg.source}'")
        df = df.loc[:, data_cfg.columns]
    return df


def _resolve_layer_dataframe(
    layer: LayerConfig,
    plot_df: pd.DataFrame,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if layer.data is None:
        layer_df = plot_df.copy()
    else:
        layer_df = _materialize_data(layer.data, data_registry)

    if layer.transforms:
        layer_df = _apply_transforms(layer_df, layer.transforms)
    return layer_df


def _apply_filters(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    result = df.copy()
    for expr in filters:
        norm_expr = _normalize_expression(expr)
        result = result.query(norm_expr, engine="python", local_dict=_CALC_ENV)
    return result


def _apply_transforms(df: pd.DataFrame, transforms: list[TransformConfig]) -> pd.DataFrame:
    result = df.copy()
    for transform in transforms:
        result = _apply_transform(result, transform)
    return result


def _apply_transform(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    if transform.op == "filter":
        if transform.filter is None:
            raise ConfigValidationError("Filter transforms require the `filter` expression.")
        return _apply_filters(df, [transform.filter])
    if transform.op == "calculate":
        return _apply_calculate(df, transform)
    if transform.op == "aggregate":
        return _apply_aggregate(df, transform)
    if transform.op == "bin":
        return _apply_bin(df, transform)
    if transform.op == "window":
        return _apply_window(df, transform)
    if transform.op == "fold":
        return _apply_fold(df, transform)
    if transform.op == "pivot":
        raise ConfigValidationError("Pivot transforms are not implemented yet for the Altair renderer.")
    raise ConfigValidationError(f"Unsupported transform operation '{transform.op}'")


def _apply_calculate(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    expr = _normalize_expression(transform.expr or "")
    target = transform.as_field or ""
    if not target:
        raise ConfigValidationError("TransformConfig.as_field is required for calculate transforms")
    result = df.copy()
    result[target] = result.eval(expr, engine="python", local_dict=_CALC_ENV)
    return result


def _apply_aggregate(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    groupby = transform.groupby or []
    aggregations = transform.aggregations or {}
    if not groupby or not aggregations:
        raise ConfigValidationError("Aggregate transforms require `groupby` and `aggregations` fields.")

    agg_kwargs: dict[str, tuple[str, str]] = {}
    for alias, expr in aggregations.items():
        func, field = _parse_function_expr(expr, expected_arg=True)
        agg_kwargs[alias] = (field, func)

    grouped = df.groupby(groupby, dropna=False).agg(**agg_kwargs).reset_index()
    return grouped


def _apply_bin(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    if not transform.field or not transform.binned_as:
        raise ConfigValidationError("Bin transforms require `field` and `binned_as`.")
    bins = transform.maxbins or 10
    result = df.copy()
    result[transform.binned_as] = pd.cut(result[transform.field], bins=bins, include_lowest=True)
    return result


def _apply_window(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    if not transform.window:
        raise ConfigValidationError("Window transforms require the `window` mapping.")
    result = df.copy()
    for alias, expr in transform.window.items():
        func, field = _parse_function_expr(expr, expected_arg=True)
        if func == "rank":
            result[alias] = result[field].rank(method="average")
        elif func == "cumsum":
            result[alias] = result[field].cumsum()
        else:
            raise ConfigValidationError(f"Window function '{func}' is not supported.")
    return result


def _apply_fold(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    if not transform.fold_fields:
        raise ConfigValidationError("Fold transforms require `fold_fields`.")
    var_name, value_name = _derive_fold_names(transform.as_fields)
    return df.melt(value_vars=transform.fold_fields, var_name=var_name, value_name=value_name)


def _parse_function_expr(expr: str, expected_arg: bool) -> tuple[str, str]:
    if "(" not in expr or not expr.endswith(")"):
        raise ConfigValidationError(f"Expression '{expr}' must be of the form func(field).")
    func, rest = expr.split("(", 1)
    value = rest[:-1].strip()
    func = func.strip()
    if expected_arg and not value:
        raise ConfigValidationError(f"Expression '{expr}' must supply an argument.")
    return func, value


def _derive_fold_names(as_fields: list[str] | None) -> tuple[str, str]:
    if not as_fields:
        return "key", "value"
    if len(as_fields) == 1:
        return as_fields[0], "value"
    return as_fields[0], as_fields[1]


def _normalize_expression(expr: str) -> str:
    return expr.replace("datum.", "").strip()


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
    for channel_name in ("x", "y", "color", "size", "shape", "opacity", "row", "column"):
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
        if isinstance(cfg.sort, list):
            kwargs["sort"] = alt.Sort(cfg.sort)
        else:
            kwargs["sort"] = cfg.sort
    if cfg.scale:
        kwargs["scale"] = _scale_to_alt(cfg.scale)
    if cfg.axis and channel_name in {"x", "y", "row", "column"}:
        kwargs["axis"] = _axis_to_alt(cfg.axis)
    if cfg.legend and channel_name in {"color", "size", "shape", "opacity"}:
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
    kwargs: dict[str, Any] = {"name": cfg.name}
    if cfg.encodings is not None:
        kwargs["encodings"] = cfg.encodings  # type: ignore[assignment]
    if cfg.fields is not None:
        kwargs["fields"] = cfg.fields
    if cfg.bind is not None:
        kwargs["bind"] = cfg.bind  # type: ignore[assignment]

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
        chart.background = background
    if guides.labels:
        LOGGER.info("Plot-level labels are not yet implemented for Altair; skipping %s labels.", len(guides.labels))
    return chart


def _build_title_params(guides: PlotLevelGuideConfig):
    subtitle_lines = [text for text in (guides.subtitle, guides.caption) if text]
    if not guides.title and not subtitle_lines:
        return None
    if subtitle_lines:
        return alt.TitleParams(text=guides.title or "", subtitle=subtitle_lines)
    return guides.title
