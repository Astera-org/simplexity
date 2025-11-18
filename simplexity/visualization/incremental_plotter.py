"""Incremental plot building utilities for memory-efficient visualization.

This module provides utilities to build Plotly figures incrementally by appending
new steps to existing plots, allowing slider functionality while keeping memory bounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import PlotConfig, PlotControlsConfig


def load_or_create_figure(figure_path: str | Path) -> go.Figure:
    """Load an existing Plotly figure from disk or create a new empty figure.

    Args:
        figure_path: Path to the saved figure JSON file

    Returns:
        Loaded figure or new empty figure if file doesn't exist
    """
    figure_path = Path(figure_path)

    if figure_path.exists():
        # Use Plotly's read_json to properly deserialize
        return pio.read_json(figure_path)

    # Create new figure with empty data
    return go.Figure()


def append_step_to_figure(
    cumulative_fig: go.Figure,
    step_plot_config: PlotConfig,
    data_registry: dict[str, Any],
    logged_steps: list[int],
) -> go.Figure:
    """Append a new step's data to an existing cumulative figure and update controls.

    Args:
        cumulative_fig: Existing figure to append to
        step_plot_config: Plot configuration for the current step
        data_registry: Data registry containing current step's data
        logged_steps: List of all step numbers that have been added (including current)

    Returns:
        Updated figure with new step's traces and controls added
    """
    # Build figure for current step only
    step_fig = build_plotly_figure(step_plot_config, data_registry)

    # Clear any controls from the step figure (they're designed for single-step plots)
    # We'll add appropriate controls for the cumulative view later
    if hasattr(step_fig, "layout"):
        if hasattr(step_fig.layout, "sliders"):
            step_fig.layout.sliders = []
        if hasattr(step_fig.layout, "updatemenus"):
            step_fig.layout.updatemenus = []

    # If this is the first step, copy layout (minus controls)
    if len(cumulative_fig.data) == 0:  # type: ignore[arg-type]
        cumulative_fig.update_layout(step_fig.layout)

    # Append all traces from step figure
    for trace in step_fig.data:
        # Add step metadata to trace name if not already present
        if hasattr(trace, "name") and trace.name:
            trace.name = f"{trace.name}"
        cumulative_fig.add_trace(trace)

    # Update controls for cumulative view based on config
    cumulative_fig = _update_controls(cumulative_fig, logged_steps, step_plot_config.controls)

    return cumulative_fig


def _update_controls(
    fig: go.Figure, steps: list[int], controls: PlotControlsConfig | None
) -> go.Figure:
    """Update interactive controls (slider/dropdown) based on controls config.

    Args:
        fig: Figure to update controls for
        steps: List of all step numbers that have been added
        controls: Controls configuration specifying which dimension gets slider/dropdown

    Returns:
        Figure with appropriate controls added
    """
    if not steps or len(fig.data) == 0:  # type: ignore[arg-type]
        return fig

    # Clear any existing controls
    fig.update_layout(sliders=[], updatemenus=[])

    # If no controls config provided, default to step slider
    if controls is None:
        return _create_step_slider_for_spatial_view(fig, steps)

    # Apply controls based on config
    if controls.slider and controls.slider.dimension == "step":
        fig = _create_step_slider_for_spatial_view(fig, steps)

    if controls.dropdown and controls.dropdown.dimension == "layer":
        fig = _create_layer_dropdown_for_temporal_view(fig, steps)

    return fig


def create_step_slider(
    fig: go.Figure, steps: list[int], controls: PlotControlsConfig | None = None
) -> go.Figure:
    """Add interactive controls (slider/dropdown) based on controls config.

    Args:
        fig: Figure to add controls to
        steps: List of step numbers that have been added
        controls: Controls configuration specifying which dimension gets slider/dropdown

    Returns:
        Figure with appropriate controls added
    """
    if not steps or len(fig.data) == 0:  # type: ignore[arg-type]
        return fig

    # If no controls config provided, default to step slider
    if controls is None:
        return _create_step_slider_for_spatial_view(fig, steps)

    # Apply controls based on config
    if controls.slider and controls.slider.dimension == "step":
        fig = _create_step_slider_for_spatial_view(fig, steps)

    if controls.dropdown and controls.dropdown.dimension == "layer":
        fig = _create_layer_dropdown_for_temporal_view(fig, steps)

    return fig


def _create_step_slider_for_spatial_view(fig: go.Figure, steps: list[int]) -> go.Figure:
    """Create step slider for spatial view (shows all layers at selected step).

    Args:
        fig: Figure to add slider to
        steps: List of step numbers

    Returns:
        Figure with step slider added
    """
    total_traces = len(fig.data)  # type: ignore[arg-type]
    num_steps = len(steps)

    if num_steps == 0:
        return fig

    traces_per_step = total_traces // num_steps

    # Clean up trace names - remove step prefix for spatial view
    for _i, trace in enumerate(fig.data):
        if hasattr(trace, "name") and trace.name and "step_" in trace.name:  # type: ignore[attr-defined]
            # Remove "step_X__" prefix, keep only layer name
            parts = trace.name.split("__")  # type: ignore[attr-defined]
            layer_parts = [p for p in parts if not p.startswith("step_")]
            if layer_parts:
                # Extract just the layer name (e.g., "layer_X" → "X")
                if layer_parts[0].startswith("layer_"):
                    layer_name = layer_parts[0].replace("layer_", "")
                else:
                    layer_name = layer_parts[0]
                trace.name = layer_name  # type: ignore[attr-defined]

    # Create slider steps with showlegend control
    slider_steps = []
    for i, step_num in enumerate(steps):
        visible = [(j >= i * traces_per_step and j < (i + 1) * traces_per_step) for j in range(total_traces)]
        step_dict = {
            "args": [
                {
                    "visible": visible,
                    "showlegend": visible,  # Only show legend for visible traces
                }
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

    # Set initial visibility and legend (show only last step)
    for i, trace in enumerate(fig.data):
        is_visible = i >= total_traces - traces_per_step
        trace.visible = is_visible  # type: ignore[attr-defined]
        trace.showlegend = is_visible  # type: ignore[attr-defined]

    return fig


def _create_layer_dropdown_for_temporal_view(fig: go.Figure, steps: list[int]) -> go.Figure:
    """Create layer dropdown for temporal view (shows all steps for selected layer).

    Args:
        fig: Figure to add dropdown to
        steps: List of step numbers (used to identify which traces belong to which layer)

    Returns:
        Figure with layer dropdown added
    """
    # Parse trace names to extract layer information
    layer_to_traces: dict[str, list[int]] = {}

    for trace_idx, trace in enumerate(fig.data):
        if hasattr(trace, "name") and trace.name and "__" in trace.name:  # type: ignore[attr-defined]
            # Extract layer name from "layer_{name}__step_{n}"
            parts = trace.name.split("__")  # type: ignore[attr-defined]
            for part in parts:
                if part.startswith("layer_"):
                    layer_name = part.replace("layer_", "")
                    if layer_name not in layer_to_traces:
                        layer_to_traces[layer_name] = []
                    layer_to_traces[layer_name].append(trace_idx)
                    break

    if not layer_to_traces:
        return fig

    # Create dropdown buttons for each layer
    buttons = []
    layers_sorted = sorted(layer_to_traces.keys())

    for layer_name in layers_sorted:
        # Create visibility array: show all traces for this layer
        visible = [i in layer_to_traces[layer_name] for i in range(len(fig.data))]  # type: ignore[arg-type]

        button = {
            "method": "update",
            "args": [{"visible": visible}],
            "label": layer_name,
        }
        buttons.append(button)

    # Add dropdown to layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ]
    )

    # Set initial visibility (first layer)
    if layers_sorted:
        first_layer_traces = layer_to_traces[layers_sorted[0]]
        for i, trace in enumerate(fig.data):
            trace.visible = i in first_layer_traces  # type: ignore[attr-defined]

    return fig


def save_figure(fig: go.Figure, figure_path: str | Path) -> None:
    """Save a Plotly figure to disk as JSON.

    Args:
        fig: Figure to save
        figure_path: Path where to save the figure
    """
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    # Use Plotly's write_json to handle numpy/JAX array serialization
    fig.write_json(figure_path)


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
