"""Incremental plot building utilities for memory-efficient visualization.

This module provides utilities to build Plotly figures incrementally by appending
new steps to existing plots, allowing slider functionality while keeping memory bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import PlotConfig


def load_or_create_figure(figure_path: str | Path) -> go.Figure:
    """Load an existing Plotly figure from disk or create a new empty figure.

    Args:
        figure_path: Path to the saved figure JSON file

    Returns:
        Loaded figure or new empty figure if file doesn't exist
    """
    figure_path = Path(figure_path)

    if figure_path.exists():
        with open(figure_path) as f:
            fig_dict = json.load(f)
        return go.Figure(fig_dict)

    # Create new figure with empty data
    return go.Figure()


def append_step_to_figure(
    cumulative_fig: go.Figure,
    step_plot_config: PlotConfig,
    data_registry: dict[str, Any],
    current_step: int,
) -> go.Figure:
    """Append a new step's data to an existing cumulative figure.

    Args:
        cumulative_fig: Existing figure to append to
        step_plot_config: Plot configuration for the current step
        data_registry: Data registry containing current step's data
        current_step: Current training step number

    Returns:
        Updated figure with new step's traces added
    """
    # Build figure for current step only
    step_fig = build_plotly_figure(step_plot_config, data_registry)

    # If this is the first step, copy layout
    if len(cumulative_fig.data) == 0:  # type: ignore[arg-type]
        cumulative_fig.update_layout(step_fig.layout)

    # Append all traces from step figure
    for trace in step_fig.data:
        # Add step metadata to trace name if not already present
        if hasattr(trace, "name") and trace.name:
            trace.name = f"{trace.name}"
        cumulative_fig.add_trace(trace)

    return cumulative_fig


def create_step_slider(fig: go.Figure, steps: list[int], layer_names: list[str] | None = None) -> go.Figure:
    """Add or update a step slider to a figure.

    Args:
        fig: Figure to add slider to
        steps: List of step numbers that have been added
        layer_names: Optional list of layer names for filtering

    Returns:
        Figure with slider added/updated
    """
    if not steps:
        return fig

    # Determine traces per step (assumes uniform structure)
    total_traces = len(fig.data)  # type: ignore[arg-type]
    num_steps = len(steps)

    if num_steps == 0:
        return fig

    traces_per_step = total_traces // num_steps

    # Create slider steps
    slider_steps = []
    for i, step_num in enumerate(steps):
        step_dict = {
            "args": [
                {"visible": [(j >= i * traces_per_step and j < (i + 1) * traces_per_step) for j in range(total_traces)]}
            ],
            "label": f"Step {step_num}",
            "method": "update",
        }
        slider_steps.append(step_dict)

    # Add slider to layout
    sliders = [
        {
            "active": len(steps) - 1,  # Show most recent step by default
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Training Step: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "steps": slider_steps,
        }
    ]

    fig.update_layout(sliders=sliders)

    # Set initial visibility (show only last step)
    for i, trace in enumerate(fig.data):
        trace.visible = i >= total_traces - traces_per_step  # type: ignore[attr-defined]

    return fig


def save_figure(fig: go.Figure, figure_path: str | Path) -> None:
    """Save a Plotly figure to disk as JSON.

    Args:
        fig: Figure to save
        figure_path: Path where to save the figure
    """
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    with open(figure_path, "w") as f:
        json.dump(fig.to_dict(), f)


def save_figure_html(fig: go.Figure, html_path: str | Path) -> None:
    """Save a Plotly figure as interactive HTML.

    Args:
        fig: Figure to save
        html_path: Path where to save the HTML file
    """
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path)


__all__ = [
    "load_or_create_figure",
    "append_step_to_figure",
    "create_step_slider",
    "save_figure",
    "save_figure_html",
]
