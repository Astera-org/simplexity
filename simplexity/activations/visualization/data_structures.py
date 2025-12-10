"""Core data structures for activation visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from simplexity.visualization.structured_configs import PlotConfig


@dataclass
class PreparedMetadata:
    """Metadata derived during activation preprocessing."""

    sequences: list[tuple[int, ...]]
    steps: np.ndarray
    select_last_token: bool


@dataclass
class ActivationVisualizationPayload:
    """Rendered visualization plus auxiliary metadata."""

    analysis: str
    name: str
    backend: str
    figure: Any
    dataframe: pd.DataFrame
    controls: VisualizationControlsState | None
    plot_config: PlotConfig


@dataclass
class VisualizationControlDetail:
    """Runtime metadata for a single control."""

    type: str
    field: str
    options: list[Any]
    cumulative: bool | None = None


@dataclass
class VisualizationControlsState:
    """Collection of optional control metadata."""

    slider: VisualizationControlDetail | None = None
    dropdown: VisualizationControlDetail | None = None
    toggle: VisualizationControlDetail | None = None
    accumulate_steps: bool = False


_SCALAR_INDEX_SENTINEL = "__SCALAR_INDEX_SENTINEL__"

__all__ = [
    "ActivationVisualizationPayload",
    "PreparedMetadata",
    "VisualizationControlDetail",
    "VisualizationControlsState",
    "_SCALAR_INDEX_SENTINEL",
]
