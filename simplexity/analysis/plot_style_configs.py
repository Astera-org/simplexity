"""Styling configuration dataclasses for analysis plots.

These configs define the declarative (YAML-configurable) parts of analysis plots,
while the dynamic layer generation remains programmatic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BasePlotStyleConfig:
    """Common styling options for all analysis plots."""

    title: str = "Analysis Plot"
    width: int = 800
    height: int = 600
    background: str | None = None
    opacity: float = 0.6


@dataclass
class PCA2DStyleConfig(BasePlotStyleConfig):
    """Styling configuration for 2D PCA plots."""

    title: str = "PCA 2D Projection"
    pc1_title: str = "PC 1"
    pc2_title: str = "PC 2"
    colorscale: str = "Viridis"
    marker_size: int | None = None


@dataclass
class PCA3DStyleConfig(BasePlotStyleConfig):
    """Styling configuration for 3D PCA plots."""

    title: str = "PCA 3D Projection"
    pc1_title: str = "PC 1"
    pc2_title: str = "PC 2"
    pc3_title: str = "PC 3"
    marker_size: int | None = None


@dataclass
class RegressionStyleConfig(BasePlotStyleConfig):
    """Styling configuration for regression simplex projection plots."""

    title: str = "Regression Simplex Projection"
    x_title: str = "X"
    y_title: str = "Y"
    true_color: str = "blue"
    predicted_color: str = "red"
    marker_size: int | None = None


@dataclass
class VarianceStyleConfig(BasePlotStyleConfig):
    """Styling configuration for variance explained plots."""

    title: str = "Cumulative Variance Explained"
    width: int = 700
    height: int = 500
    component_title: str = "Components"
    variance_title: str = "Cumulative Variance"
    line_width: int = 2
    axis_format: str = ".0%"
    max_components: int | None = 20
    variance_thresholds: list[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    show_threshold_lines: bool = False


__all__ = [
    "BasePlotStyleConfig",
    "PCA2DStyleConfig",
    "PCA3DStyleConfig",
    "RegressionStyleConfig",
    "VarianceStyleConfig",
]
