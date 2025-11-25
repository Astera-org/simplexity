"""Structured visualization configuration dataclasses.

This module implements the schema described in docs/visualization.md. The
dataclasses are intentionally backend-agnostic so that Hydra configs can be
validated once and rendered by different visualization engines (Altair,
plotnine, matplotlib, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simplexity.exceptions import ConfigValidationError


def _ensure(condition: bool, message: str) -> None:
    """Raise ConfigValidationError if condition is not met."""
    if not condition:
        raise ConfigValidationError(message)


@dataclass
class DataConfig:
    """Specifies the logical data source and lightweight filtering."""

    source: str = "main"
    filters: list[str] = field(default_factory=list)
    columns: list[str] | None = None


@dataclass
class TransformConfig:  # pylint: disable=too-many-instance-attributes
    """Represents a single data transform stage."""

    op: str  # ["filter", "calculate", "aggregate", "bin", "window", "fold", "pivot"]
    filter: str | None = None
    as_field: str | None = None
    expr: str | None = None
    groupby: list[str] | None = None
    aggregations: dict[str, str] | None = None
    field: str | None = None
    binned_as: str | None = None
    maxbins: int | None = None
    window: dict[str, str] | None = None
    frame: list[int | None] | None = None
    fold_fields: list[str] | None = None
    as_fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.op == "filter":
            _ensure(bool(self.filter), "TransformConfig.filter must be provided when op='filter'")
        if self.op == "calculate":
            _ensure(bool(self.as_field), "TransformConfig.as_field is required for calculate transforms")
            _ensure(bool(self.expr), "TransformConfig.expr is required for calculate transforms")
        if self.op == "aggregate":
            _ensure(bool(self.groupby), "TransformConfig.groupby is required for aggregate transforms")
            _ensure(
                bool(self.aggregations),
                "TransformConfig.aggregations is required for aggregate transforms",
            )
        if self.op == "bin":
            _ensure(bool(self.field), "TransformConfig.field is required for bin transforms")
            _ensure(bool(self.binned_as), "TransformConfig.binned_as is required for bin transforms")
        if self.op == "window":
            _ensure(bool(self.window), "TransformConfig.window is required for window transforms")


@dataclass
class ScaleConfig:
    """Describes how raw data values are mapped to visual ranges."""

    type: str | None = None  # ["linear", "log", "sqrt", "pow", "symlog", "time", "utc", "ordinal", "band", "point"]
    domain: list[Any] | None = None
    range: list[Any] | None = None
    clamp: bool | None = None
    nice: bool | None = None
    reverse: bool | None = None


@dataclass
class AxisConfig:
    """Axis settings for positional channels."""

    title: str | None = None
    grid: bool | None = None
    format: str | None = None
    tick_count: int | None = None
    label_angle: float | None = None
    visible: bool = True


@dataclass
class LegendConfig:
    """Legend settings for categorical or continuous mappings."""

    title: str | None = None
    orient: str | None = None
    visible: bool = True


@dataclass
class ChannelAestheticsConfig:  # pylint: disable=too-many-instance-attributes
    """Represents one visual encoding channel (x, y, color, etc.)."""

    field: str | None = None
    type: str | None = None  # ["quantitative", "ordinal", "nominal", "temporal"]
    value: Any | None = None
    aggregate: str | None = None
    bin: bool | None = None
    time_unit: str | None = None
    scale: ScaleConfig | None = None
    axis: AxisConfig | None = None
    legend: LegendConfig | None = None
    sort: str | list[Any] | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        if self.field is not None and self.value is not None:
            raise ConfigValidationError(
                "ChannelAestheticsConfig cannot specify both 'field' and 'value'; prefer 'field'."
            )


@dataclass
class AestheticsConfig:  # pylint: disable=too-many-instance-attributes
    """Collection of channel encodings for a layer."""

    x: ChannelAestheticsConfig | None = None
    y: ChannelAestheticsConfig | None = None
    z: ChannelAestheticsConfig | None = None
    color: ChannelAestheticsConfig | None = None
    size: ChannelAestheticsConfig | None = None
    shape: ChannelAestheticsConfig | None = None
    opacity: ChannelAestheticsConfig | None = None
    tooltip: list[ChannelAestheticsConfig] | None = None
    row: ChannelAestheticsConfig | None = None
    column: ChannelAestheticsConfig | None = None
    detail: ChannelAestheticsConfig | None = None


@dataclass
class GeometryConfig:
    """Visual primitive used to draw the layer."""

    type: str  # [point, line, area, bar, rect, rule, tick, circle, square, text, boxplot, errorbar, errorband]
    props: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure(isinstance(self.props, dict), "GeometryConfig.props must be a dictionary")


@dataclass
class SelectionConfig:
    """Interactive selection definition."""

    name: str
    type: str = "interval"  # ["interval", "single", "multi"]
    encodings: list[str] | None = None
    fields: list[str] | None = None
    bind: dict[str, Any] | None = None


@dataclass
class PlotSizeConfig:
    """Size and layout metadata for an entire plot."""

    width: int | None = None
    height: int | None = None
    autosize: str | None = None


@dataclass
class LabelConfig:
    """Free-form labels or annotations."""

    text: str | None = None
    x: float | str | None = None
    y: float | str | None = None
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlotLevelGuideConfig:
    """Titles and caption level guides."""

    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    labels: list[LabelConfig] | None = None


@dataclass
class FacetConfig:
    """High-level faceting instructions."""

    row: str | None = None
    column: str | None = None
    wrap: int | None = None


@dataclass
class LayerConfig:
    """A single layer in a composed plot."""

    name: str | None = None
    data: DataConfig | None = None
    transforms: list[TransformConfig] = field(default_factory=list)
    geometry: GeometryConfig = field(default_factory=lambda: GeometryConfig(type="point"))
    aesthetics: AestheticsConfig = field(default_factory=AestheticsConfig)
    selections: list[SelectionConfig] = field(default_factory=list)


@dataclass
class PlotConfig:  # pylint: disable=too-many-instance-attributes
    """Top-level configuration for one plot."""

    backend: str = "altair"  # ["altair", "plotly"]
    data: DataConfig = field(default_factory=DataConfig)
    transforms: list[TransformConfig] = field(default_factory=list)
    layers: list[LayerConfig] = field(default_factory=list)
    facet: FacetConfig | None = None
    size: PlotSizeConfig = field(default_factory=PlotSizeConfig)
    guides: PlotLevelGuideConfig = field(default_factory=PlotLevelGuideConfig)
    background: str | None = None
    selections: list[SelectionConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        _ensure(self.layers is not None, "PlotConfig.layers must be a list (can be empty)")


@dataclass
class GraphicsConfig:
    """Root Visualization config that multiplexes multiple named plots."""

    default_backend: str = "altair"  # ["altair", "plotly"]
    plots: dict[str, PlotConfig] = field(default_factory=dict)
