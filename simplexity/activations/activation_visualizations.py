"""Helpers for building activation visualizations from analysis outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from simplexity.activations.visualization.data_structures import (
    _SCALAR_INDEX_SENTINEL,
    ActivationVisualizationPayload,
    PreparedMetadata,
    VisualizationControlDetail,
    VisualizationControlsState,
)
from simplexity.activations.visualization.dataframe_builders import (
    _build_dataframe,
    _build_metadata_columns,
)
from simplexity.activations.visualization.preprocessing import _apply_preprocessing
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationControlsConfig,
)
from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import PlotConfig


def _parse_scalar_expression(expr: str) -> tuple[str, str | None]:
    """Parse a scalar expression that may contain an aggregation function.

    Args:
        expr: Expression like "layer_0_rmse" or "min(layer_0_rmse)"

    Returns:
        Tuple of (scalar_key, aggregation_function or None)
    """
    expr = expr.strip()
    agg_match = re.match(r"^(min|max|avg|mean|latest|first|last)\((.+)\)$", expr)
    if agg_match:
        agg_func = agg_match.group(1)
        scalar_key = agg_match.group(2).strip()
        return (scalar_key, agg_func)
    return (expr, None)


def _compute_aggregation(
    history: list[tuple[int, float]],
    agg_func: str,
) -> float:
    """Compute aggregation over scalar history.

    Args:
        history: List of (step, value) tuples
        agg_func: Aggregation function name (min, max, avg, mean, latest, first, last)

    Returns:
        Aggregated value
    """
    if not history:
        raise ConfigValidationError(f"Cannot compute {agg_func} over empty history")

    values = [value for _, value in history]

    if agg_func == "min":
        return float(np.min(values))
    elif agg_func == "max":
        return float(np.max(values))
    elif agg_func in ("avg", "mean"):
        return float(np.mean(values))
    elif agg_func in ("latest", "last"):
        return history[-1][1]
    elif agg_func == "first":
        return history[0][1]
    else:
        raise ConfigValidationError(f"Unknown aggregation function: {agg_func}")


def _render_title_template(
    title: str | None,
    title_scalars: dict[str, str] | None,
    scalars: Mapping[str, float],
    scalar_history: Mapping[str, list[tuple[int, float]]],
) -> str | None:
    """Render a title template by substituting scalar values and aggregations.

    Args:
        title: Title string potentially containing format placeholders like {rmse:.3f}
        title_scalars: Mapping from template variable names to scalar keys or expressions
        scalars: Available current scalar values
        scalar_history: Historical scalar values for aggregations

    Returns:
        Rendered title string with scalar values substituted, or None if title is None

    Examples:
        title_scalars: {"rmse": "layer_0_rmse", "best": "min(layer_0_rmse)"}
        This will substitute {rmse} with current value and {best} with historical minimum.
    """
    if title is None:
        return None

    if title_scalars is None or not title_scalars:
        return title

    scalar_values = {}
    for var_name, scalar_expr in title_scalars.items():
        scalar_key, agg_func = _parse_scalar_expression(scalar_expr)

        if agg_func is None:
            # No aggregation, use current value
            if scalar_key in scalars:
                scalar_values[var_name] = scalars[scalar_key]
            else:
                raise ConfigValidationError(
                    f"Title template references scalar '{scalar_key}' (var: '{var_name}') but it is not available. "
                    f"Available scalars: {list(scalars.keys())}"
                )
        else:
            # Aggregation requested, use history
            if scalar_key not in scalar_history:
                raise ConfigValidationError(
                    f"Title template requests {agg_func}({scalar_key}) but no history available for '{scalar_key}'. "
                    f"Available history keys: {list(scalar_history.keys())}"
                )
            history = scalar_history[scalar_key]
            scalar_values[var_name] = _compute_aggregation(history, agg_func)

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
    scalar_history: Mapping[str, list[tuple[int, float]]],
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
            scalar_history,
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
                scalar_history,
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
    """Render a visualization figure from plot configuration and dataframe."""
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
    # Filter out "_no_layer_" placeholder used for layer-independent data (e.g., ground truth)
    if field == "layer":
        options = [opt for opt in options if opt != "_no_layer_"]
    return VisualizationControlDetail(type=control_type, field=field, options=options, cumulative=cumulative)


__all__ = [
    "ActivationVisualizationPayload",
    "PreparedMetadata",
    "VisualizationControlDetail",
    "VisualizationControlsState",
    "_SCALAR_INDEX_SENTINEL",
    "build_visualization_payloads",
    "render_visualization",
]
