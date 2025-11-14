"""Reusable helpers for preparing data prior to rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_registry import DataRegistry, resolve_data_source
from simplexity.visualization.structured_configs import (
    DataConfig,
    LayerConfig,
    TransformConfig,
)

CALC_ENV = {
    "np": np,
    "pd": pd,
    "math": math,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "clip": np.clip,
}


def normalize_expression(expr: str) -> str:
    """Normalize expressions shared between pandas and Vega-Lite syntaxes."""
    return expr.replace("datum.", "").strip()


def materialize_data(data_cfg: DataConfig, data_registry: DataRegistry | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve a logical data source and apply lightweight filters/column selection."""
    df = resolve_data_source(data_cfg.source, data_registry).copy()
    if data_cfg.filters:
        df = apply_filters(df, data_cfg.filters)
    if data_cfg.columns:
        missing = [col for col in data_cfg.columns if col not in df.columns]
        if missing:
            raise ConfigValidationError(f"Columns {missing} are not present in data source '{data_cfg.source}'")
        df = df.loc[:, data_cfg.columns]
    return df


def build_plot_level_dataframe(
    data_cfg: DataConfig,
    transforms: list[TransformConfig],
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Materialize the base dataframe for a plot, applying plot-level transforms."""
    df = materialize_data(data_cfg, data_registry)
    return apply_transforms(df, transforms)


def resolve_layer_dataframe(
    layer: LayerConfig,
    plot_df: pd.DataFrame,
    data_registry: DataRegistry | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Resolve the dataframe for an individual layer."""
    if layer.data is None:
        df = plot_df.copy()
    else:
        df = materialize_data(layer.data, data_registry)
    if layer.transforms:
        df = apply_transforms(df, layer.transforms)
    return df


def apply_filters(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    """Apply pandas-compatible query filters."""
    result = df.copy()
    for expr in filters:
        norm_expr = normalize_expression(expr)
        result = result.query(norm_expr, engine="python", local_dict=CALC_ENV)
    return result


def apply_transforms(df: pd.DataFrame, transforms: list[TransformConfig]) -> pd.DataFrame:
    """Sequentially apply configured transforms to a dataframe."""
    result = df.copy()
    for transform in transforms:
        result = _apply_transform(result, transform)
    return result


def _apply_transform(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    if transform.op == "filter":
        if transform.filter is None:
            raise ConfigValidationError("Filter transforms require the `filter` expression.")
        return apply_filters(df, [transform.filter])
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
        raise ConfigValidationError("Pivot transforms are not implemented yet.")
    raise ConfigValidationError(f"Unsupported transform operation '{transform.op}'")


def _apply_calculate(df: pd.DataFrame, transform: TransformConfig) -> pd.DataFrame:
    expr = normalize_expression(transform.expr or "")
    target = transform.as_field or ""
    if not target:
        raise ConfigValidationError("TransformConfig.as_field is required for calculate transforms")
    result = df.copy()
    result[target] = result.eval(expr, engine="python", local_dict=CALC_ENV)
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


__all__ = [
    "CALC_ENV",
    "apply_filters",
    "apply_transforms",
    "build_plot_level_dataframe",
    "materialize_data",
    "normalize_expression",
    "resolve_layer_dataframe",
]
