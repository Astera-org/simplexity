# 📊 Visualization Configuration System — Design Document

**Status:** Draft

---

## 0. Background

Want to be able to generate graphics in a way fully configurable via composable yaml configs following the structure of Leland Wilkinson’s _Grammar of Graphics_

1. Data
2. Transformations / Statistics
3. Scales
4. Coordinates
5. Aesthetics
6. Geometries
7. Guides
8. Facets / Layout
9. Layering

The design is supposed to be agnostic to the visualization backend, but `Altair` and `plotnine` in particular map closely to this structure:

### Altair

- data
- transform
- scale
- (coordinates isn't exposed as strongly, only geoshapes/polar-ish transforms)
- encoding (aesthetics)
- mark (geoms)
- axis/legend (guides)
- facet
- selection
- repeat
- resolve

Notice that Altair does not distinguish “Aesthetic” vs “Geometry” vs “Scaling” in separate top-level structures—they’re all parameters to:

```python
Chart(...)
    .transform_*
    .encode(...)
    .mark_*
    .properties(...)
    .facet(...)
    .layer(...)
```

### ggplot2

- data
- statistics
- scales
- coordinates
- aesthetics
- geometries
- theme (guides are considered part of geometry or theme)
- facets
- (layering is implicit, `+` adds a layer)

## 1. Overview

This document describes the design of a **backend-agnostic, declarative visualization configuration system** used to generate both **interactive** and **static** plots for experiment analysis.

Configurations are written in **YAML**, validated/structured using **Hydra**, and rendered into visualizations by pluggable **backend renderers**.

The first supported visualization backend is **Altair** (Vega-Lite), chosen for its powerful interactivity and declarative JSON-based grammar. The system is designed so that future backends such as **plotnine**, **matplotlib**, or **plotly** can be added without modifying existing configs.

---

## 2. Goals

### 2.1 Primary Goals

- **Declarative Specs**
  Make plots fully described by YAML config files (Hydra) rather than imperative code.

- **Backend Agnostic**
  A single plot config should be renderable in:

  - Altair (interactive plots)
  - Plotnine (static publication-quality)
  - Matplotlib (fallback / custom needs)
  - Plotly (potential future)

- **Consistent Grammar**
  Model the configuration language after a unified Grammar of Graphics:

  - data → transforms → aesthetics → geometry → guides → facets.

- **Interactive + Static Support**
  Altair used for rich interactive visualizations; other backends can target static use cases.

- **Reproducibility**
  Configs should be versionable, diffable, and stable across experiments.

### 2.2 Non-Goals

- Implementing 100% of Vega-Lite, ggplot2, or matplotlib features.
- Creating a one-to-one mapping of primitives across backends.
- Allowing arbitrary Python code inside configs.

---

## 3. Key Requirements

### Functional Requirements

- Ability to select and transform DataFrame sources declaratively.
- Layered graphics (multiple geometry on the same plot).
- Aesthetic mapping (x, y, color, size, shape, opacity).
- Support for:

  - transformations (filter, calculate, aggregate, bin, window)
  - geometry (point, line, bar, area, etc.)
  - scales (domain/range, log/linear, etc.)
  - axes and legends
  - facets (row, column, wrap)
  - selections (Altair interactivity)
  - tooltips

- Backend routing based on config (`backend: altair`).

### Non-Functional Requirements

- Config schema should be stable and extensible.
- Zero runtime dependency on Altair for non-Altair backends.
- Models should not assume a specific visualization implementation.

---

## 4. High-Level Architecture

```
         +-------------------+
         |    YAML Config    |
         +---------+---------+
                   |
                   v
       +------------------------+
       |  Hydra Structured cfg  |
       |    (PlotConfig, ...)   |
       +-----------+------------+
                   |
                   v
      +---------------------------+
      |   Backend Renderer API    |
      |  build_plot(plot_cfg, df) |
      +-----+-----------+---------+
            |           |
   +--------v--+     +--v---------+
   | Altair    |     | Plotnine   |   (future)
   | Renderer  |     | Renderer   |
   +-----------+     +------------+
```

The **PlotConfig** object serves as the canonical intermediate representation (IR).
Each backend renderer consumes the IR and constructs native visualization objects.

---

## 5. Schema Design

The schema is defined using Python `@dataclass`es to support Hydra structured configs.

### 5.1 Top-level

```python
from dataclasses import dataclass, field
from typing import Any, Literal

BackendType = Literal["altair", "plotnine", "matplotlib", "plotly"]

@dataclass
class GraphicsConfig:
    """
    Root config object you can use as your Hydra config.

    You can either:
      - use fixed attributes like plot_1, plot_2, ...
      - or a dict of named plots.
    Here we choose a dict for flexibility.
    """
    default_backend: BackendType = "altair"

    # Named plots, e.g. {"loss_over_time": PlotConfig(...), "accuracy_hist": ...}
    plots: dict[str, PlotConfig] = field(default_factory=dict)
```

Each entry in `plots` is a named visualization.

---

## 5.2 PlotConfig

```python
@dataclass
class PlotConfig:
    """
    Top-level configuration for a single plot.
    """
    backend: BackendType = "altair"

    # Data + transforms
    data: DataConfig = field(default_factory=DataConfig)
    transforms: list[TransformConfig] = field(default_factory=list)

    # Visual structure
    layers: list[LayerConfig] = field(default_factory=list)
    facet: FacetConfig | None = None

    # Global sizing & guides
    size: PlotSizeConfig = field(default_factory=PlotSizeConfig)
    guides: PlotLevelGuideConfig = field(default_factory=PlotLevelGuideConfig)

    # Background & theme-ish options
    background: str | None = None  # e.g. "#ffffff"

    # Global selections (e.g. selection that multiple layers reference).
    # Layer-level selections can reference plot-level selections by name.
    # Execution: plot-level selections are defined first, then layer selections.
    selections: list[SelectionConfig] = field(default_factory=list)
```

This matches core Grammar-of-Graphics components:

- **data → transformations → layers → facet → guides**.

---

## 5.3 Data

Which dataset(s) are used.

```python
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    """
    Specifies which DataFrame to use and how to subset it.

    This is backend-agnostic: your code just needs to know how to resolve
    `source` to an actual pandas.DataFrame.
    """
    source: str = "main"      # logical name of a DataFrame
    filters: list[str] = field(default_factory=list)
    # e.g. ["split == 'train'", "loss < 2.0"]
    #
    # You can interpret these as pandas.query expressions,
    # or later adapt them to Vega-Lite filter expressions.

    # Optional subset of columns to keep
    columns: list[str] | None = None
```

Includes dataset selection + subsetting.

**Filter Expression Language:**
Filters use a unified expression syntax that backends interpret appropriately:

- For pandas-based backends (plotnine, matplotlib): interpreted as pandas `.query()` expressions
- For Vega-Lite (Altair): converted to Vega-Lite filter expressions
- Expression format: Python-like syntax (e.g., `"split == 'train'"`, `"loss < 2.0"`)

**Data Source Resolution:**
The `source` field references a logical name that must be resolved to an actual `pandas.DataFrame` via a data registry (see Section 5.13).

---

## 5.4 Transformations / Statistics

Filtering, binning, aggregation, windowing, derived calculations.

Transform operations form a pipeline. Examples include:

```python
from dataclasses import dataclass, field
from typing import Any, Literal

TransformOp = Literal[
    "filter",       # keep rows where expression is true
    "calculate",    # create or overwrite a field from an expression
    "aggregate",    # groupby + aggregation
    "bin",          # numeric binning
    "window",       # window functions (rank, rolling, etc.)
    "fold",         # wide → long
    "pivot",        # long → wide
]

@dataclass
class TransformConfig:
    """
    A single transformation step in the pipeline.

    This is a superset of what both Vega-Lite and pandas can do; each backend
    can choose which transforms to support.
    """
    op: TransformOp

    # For "filter": an expression string.
    # Expression syntax: Python-like (e.g., "loss < 1.0", "split == 'train'").
    # Backends interpret: pandas uses .query(), Vega-Lite converts to filter expressions.
    filter: str | None = None

    # For "calculate": new field and expression.
    # Expression syntax: Python-like (e.g., "log(datum.loss)" for Vega-Lite,
    # "log(loss)" for pandas). Backends convert as needed.
    as_field: str | None = None
    expr: str | None = None

    # For "aggregate": groupby fields + aggregations
    groupby: list[str] | None = None
    aggregations: dict[str, str] | None = None
    # e.g. {"mean_loss": "mean(loss)", "max_acc": "max(accuracy)"}

    # For "bin": field to bin and new field name
    field: str | None = None
    binned_as: str | None = None
    maxbins: int | None = None

    # For "window": window calculations (rank, rolling, etc.)
    window: dict[str, str] | None = None
    # e.g. {"rank_loss": "rank(loss)"}
    frame: list[int | None] | None = None  # e.g. [-1, 1]

    # For "fold" / "pivot": wide/long reshaping
    fold_fields: list[str] | None = None
    as_fields: list[str] | None = None
    # etc. (you can extend as you need)
```

Only some backends will implement some operations. This is fine—unused transforms are no-ops on unsupported backends.

- Can be applied in pandas (plotnine/matplotlib)
- Or passed to Vega-Lite’s `transform_*`

---

## 5.5 Scales

Functions that map data values to perceptual ranges (linear, log, color maps).

```python
from dataclasses import dataclass, field
from typing import Any, Literal

ScaleType = Literal[
    "linear",
    "log",
    "sqrt",
    "pow",
    "symlog",
    "time",
    "utc",
    "ordinal",
    "band",
    "point",
]


@dataclass
class ScaleConfig:
    """
    Describes how data values are mapped to visual channels.

    This is generic enough to map to:
      - Altair: scale=...
      - ggplot/plotnine: scale_x_continuous, etc.
    """
    type: ScaleType | None = None
    domain: list[Any] | None = None  # e.g. [0, 1] or ["a", "b", "c"]
    range: list[Any] | None = None   # e.g. [0, 800] or color list
    clamp: bool | None = None
    nice: bool | None = None
    reverse: bool | None = None
```

Attached to each channel (aesthetics.x.scale, etc.).

---

## 5.6 Aesthetics (Encodings)

Mappings from data fields to perceptual channels: x, y, color, size, shape, opacity, etc.

`AestheticsConfig` captures x, y, color, etc.:

```python
from dataclasses import dataclass, field
from typing import Any, Literal

# Channel types: "quantitative" (numeric), "ordinal" (ordered categorical),
# "nominal" (unordered categorical), "temporal" (time/date)
ChannelType = Literal["quantitative", "ordinal", "nominal", "temporal"]

@dataclass
class ChannelAestheticsConfig:
    """
    Represents one visual channel (x, y, color, size, etc.)

    This is a generic version of a Vega-Lite "encoding" entry:
      field, type, aggregate, bin, scale, axis, legend, value (constant).
    """
    field: str | None = None
    type: ChannelType | None = None  # "quantitative", "nominal", etc.

    # Either constant value OR data field – not both at once.
    # Validation: if both are set, field takes precedence (or raise error at config validation).
    value: Any | None = None  # e.g. fixed color, size, opacity

    # Aggregation and binning
    aggregate: str | None = None  # e.g. "mean", "sum", "count"
    bin: bool | None = None
    time_unit: str | None = None   # e.g. "year", "month", etc.

    # Scale / guides
    scale: ScaleConfig | None = None
    axis: AxisConfig | None = None
    legend: LegendConfig | None = None

    # Sorting
    sort: str | list[Any] | None = None
    # e.g. "ascending", "descending", or explicit domain order


@dataclass
class AestheticsConfig:
    """
    Collection of channel aesthetics for a given layer.

    These correspond to:
      - Altair: chart.encode(x=..., y=..., color=..., tooltip=..., etc.)
      - ggplot: aes(x=..., y=..., color=..., ...)
    """
    x: ChannelAestheticsConfig | None = None
    y: ChannelAestheticsConfig | None = None
    z: ChannelAestheticsConfig | None = None

    color: ChannelAestheticsConfig | None = None
    size: ChannelAestheticsConfig | None = None
    shape: ChannelAestheticsConfig | None = None
    opacity: ChannelAestheticsConfig | None = None

    tooltip: list[ChannelAestheticsConfig] | None = None

    # Note: row/column in AestheticsConfig are deprecated in favor of
    # plot-level FacetConfig. These may be used for layer-specific faceting
    # in some backends, but FacetConfig.row/column should be preferred.
    row: ChannelAestheticsConfig | None = None
    column: ChannelAestheticsConfig | None = None
```

This structure is expressive enough for:

- Altair: `.encode(x=..., y=..., color=...)`
- ggplot2: `aes(x=..., y=..., color=...)`
- Matplotlib: parameters to `scatter`, `plot`, etc.

The optional `z` channel is ignored by strictly 2D backends, but enables 3D
scatter support for engines such as Plotly.

---

## 5.7 Geometry

Visual primitives: point, line, bar, area, text.

```python
GeometryType = Literal[
    "point",
    "line",
    "area",
    "bar",
    "rect",
    "rule",
    "tick",
    "circle",
    "square",
    "text",
    "boxplot",
    "errorbar",
    "errorband",
]

@dataclass
class GeometryConfig:
    """
    Visual primitive used for a layer.

    Backend mapping examples:
      - Altair: chart.mark_line(**mark_props)
      - plotnine: geom_line(**kwargs)
    """
    type: GeometryType
    # Arbitrary properties passed to backend's mark/geom/artist:
    # e.g. {"filled": True, "interpolate": "monotone", "strokeWidth": 2}
    props: dict[str, Any] = field(default_factory=dict)
```

Backend mapping:

- Altair: `mark_bar`, `mark_point`, …
- plotnine: `geom_bar`, `geom_point`, …

---

## 5.8 Layers

Composition rule: a plot may have multiple layers, each with its own data, aesthetics, transforms, mark, scale overrides.
Layers encapsulate their own data + aesthetics + geometry
Each layer is a visual primitive applied over the same or different data:

```python
@dataclass
class LayerConfig:
    """
    A single layer in the plot.

    Each layer can have its own data source, transforms, geometry, aesthetics,
    and (optionally) its own selections.
    """
    name: str | None = None

    # Optional per-layer data override. If None, inherit from plot.data.
    # Note: If a layer specifies its own data source, plot-level transforms
    # do NOT apply to it. Layer transforms are applied after resolving layer data.
    data: DataConfig | None = None

    # Optional transforms specific to this layer.
    # Execution order:
    #   1. Plot-level data + filters + transforms (if layer.data is None)
    #   2. Layer data override (if specified) + filters
    #   3. Layer transforms
    #   4. Geometry + aesthetics rendering
    transforms: list[TransformConfig] = field(default_factory=list)

    geometry: GeometryConfig = field(
        default_factory=lambda: GeometryConfig(type="point", props={})
    )

    aesthetics: AestheticsConfig = field(
        default_factory=AestheticsConfig
    )

    # Layer-specific selections. Can reference plot-level selections by name.
    selections: list[SelectionConfig] = field(default_factory=list)
```

Multiple layers allow:

- overlays
- aggregated + raw plots
- annotation layers
- regression layers

---

## 5.9 Facets

How multiple small plots are arranged (row, column, wrap).

```python
@dataclass
class FacetConfig:
    """
    High-level faceting.

    This can be interpreted as:
      - Altair: facet=... / row/column encoding
      - plotnine: facet_wrap / facet_grid
    """
    row: str | None = None
    column: str | None = None
    wrap: int | None = None  # for wrap-style faceting (1D grid)
```

Maps to:

- Altair: `facet()`, `row=`, `column=`
- plotnine: `facet_wrap`, `facet_grid`

---

## 5.10 Guides (Axes, Legend, Title)

Plot-level guides:

```python
@dataclass
class PlotLevelGuideConfig:
    """
    Plot-level guides and title (backend-agnostic).
    """
    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    labels: list[LabelConfig] | None = None
```

Channel guides handled by `AxisConfig` and `LegendConfig`.

```python
@dataclass
class AxisConfig:
    title: str | None = None
    grid: bool | None = None
    format: str | None = None  # e.g. ".2f", "%Y-%m-%d"
    tick_count: int | None = None
    label_angle: float | None = None
    visible: bool = True


@dataclass
class LegendConfig:
    title: str | None = None
    orient: str | None = None   # "right", "left", "top", "bottom", "none"
    # Altair: "none" disables legend; for other backends, map as appropriate
    visible: bool = True
```

```python
@dataclass
class LabelConfig:
    """
    Configuration for plot labels (annotations, text overlays, etc.).

    To be fully defined based on specific labeling needs.
    """
    text: str | None = None
    x: float | str | None = None  # position or field name
    y: float | str | None = None
    # Additional properties TBD
```

---

## 5.11 Plot Size

Plot size

```python
@dataclass
class PlotSizeConfig:
    """
    Size/layout configuration.
    """
    width: int | None = None      # pixels or backend units
    height: int | None = None
    autosize: str | None = None   # e.g. Altair/Vega-Lite autosize mode
```

---

## 5.12 Selections (Interactivity)

Selections are first-class objects in the schema:

```python
SelectionType = Literal["interval", "single", "multi"]


@dataclass
class SelectionConfig:
    """
    Abstract interactive selection.

    For Altair:
      - maps to selection_interval(), selection_single(), selection_multi()
    For other backends:
      - may be ignored, or handled by a UI layer.
    """
    name: str
    type: SelectionType = "interval"
    encodings: list[str] | None = None  # e.g. ["x", "y", "color"]
    fields: list[str] | None = None     # data fields for selection
    bind: dict[str, Any] | None = None  # UI bindings (sliders, dropdowns, etc.)
```

Backend behavior:

- Altair: maps to `selection_interval`, `selection_point`
- others: ignore or support via UI layer

---

## 5.13 Data Registry / Source Management

The visualization system requires a **data registry** that maps logical source names to actual `pandas.DataFrame` objects.

**Data Registry Interface:**

```python
from typing import Protocol

class DataRegistry(Protocol):
    """Protocol for data source resolution."""
    def get(self, source_name: str) -> pd.DataFrame:
        """Resolve a logical source name to a DataFrame."""
        ...
```

**Usage Pattern:**

```python
# Data registry is provided by the caller
data_registry = {
    "main": df_main,
    "metrics": df_metrics,
    "validation": df_val,
}

# Renderer uses registry to resolve DataConfig.source
chart = build_altair_chart(plot_cfg, data_registry)
```

**Error Handling:**

- If a source name doesn't exist in the registry, renderers should raise a `ValueError` with a clear message.
- Missing fields in DataFrames are handled at render time (backend-specific behavior).

**Lifecycle:**

- DataFrames are provided at render time, not stored in configs.
- This allows the same config to work with different datasets.
- Data registry can be populated from files, databases, or in-memory DataFrames.

---

## 6. Example YAML

```yaml
default_backend: altair

plots:
  loss_over_time:
    data:
      source: "metrics"
      filters: ["split == 'train'"]

    transforms:
      - op: calculate
        as_field: log_loss
        expr: "log(datum.loss)"

    size:
      width: 600
      height: 350

    guides:
      title: "Training Loss Over Time"

    layers:
      - name: raw_runs
        geometry:
          type: line
          props:
            opacity: 0.3
        aesthetics:
          x: { field: step, type: quantitative }
          y: { field: log_loss, type: quantitative }
          color: { field: run_id, type: nominal }

      - name: mean_line
        geometry: { type: line, props: { strokeWidth: 3 } }
        aesthetics:
          x: { field: step, type: quantitative }
          y: { field: log_loss, type: quantitative, aggregate: mean }
          color: { value: "black" }
```

**Additional Examples:**

**Example with Facets:**

```yaml
plots:
  loss_by_split:
    data:
      source: "metrics"
    facet:
      row: split
      column: model_type
    layers:
      - geometry:
          type: line
        aesthetics:
          x: { field: step, type: quantitative }
          y: { field: loss, type: quantitative }
          color: { field: run_id, type: nominal }
```

**Example with Selections:**

```yaml
plots:
  interactive_scatter:
    data:
      source: "results"
    selections:
      - name: brush
        type: interval
        encodings: [x, y]
    layers:
      - geometry:
          type: point
        aesthetics:
          x: { field: accuracy, type: quantitative }
          y: { field: loss, type: quantitative }
          color:
            field: model_type
            type: nominal
            # Selection condition would be applied here in Altair
```

**Example with Multiple Data Sources:**

```yaml
plots:
  comparison:
    layers:
      - name: training_data
        data:
          source: "train_metrics"
        geometry:
          type: line
        aesthetics:
          x: { field: epoch, type: quantitative }
          y: { field: loss, type: quantitative }

      - name: validation_data
        data:
          source: "val_metrics"
        geometry:
          type: line
          props:
            strokeDash: [5, 5]
        aesthetics:
          x: { field: epoch, type: quantitative }
          y: { field: loss, type: quantitative }
```

---

## 7. Validation & Error Handling

### 7.1 Validation Strategy

**Schema Validation:**

- Hydra automatically validates config structure against dataclass schemas.
- Type checking ensures correct types for all fields.

**Semantic Validation:**
Performed at config load time (before rendering):

- **ChannelAestheticsConfig**: `field` and `value` cannot both be set (validation error).
- **Required fields**: Certain geometries require specific aesthetics (e.g., `bar` needs `x` or `y`).
- **Transform parameters**: Validate that required parameters are present for each transform `op`.
- **Data source existence**: Check that all referenced sources exist in the data registry (at render time).

**Backend Capability Validation:**

- Warn (or error) if a config uses features unsupported by the selected backend.
- Example: Using `selections` with `backend: plotnine` would generate a warning.

### 7.2 Error Handling

**Config Load Errors:**

- Invalid YAML syntax → YAML parse error
- Missing required fields → Hydra validation error
- Invalid field values → Type/validation error

**Render-Time Errors:**

- Missing data source → `ValueError: Data source 'X' not found in registry`
- Missing DataFrame column → Backend-specific (Altair: field error, pandas: KeyError)
- Transform failure → Backend-specific error with context
- Unsupported feature → Warning logged, feature ignored (or error if critical)

**Error Messages:**
All errors should include:

- The config path/plot name where the error occurred
- The specific field or operation that failed
- Suggested fixes when possible

---

## 8. Altair Renderer (Summary)

A backend renderer converts `PlotConfig` → Altair `Chart`:

```
plot_cfg
   ↓
resolve pandas DataFrame
   ↓
apply pandas-level filters
   ↓
construct Altair base Chart
   ↓
apply Altair transforms
   ↓
construct each Layer
   ↓
combine via alt.layer(...)
   ↓
apply title, size, facet, background
   ↓
return Chart
```

Renderer API:

```python
def build_altair_chart(plot_cfg: PlotConfig,
                       data_registry: dict[str, pd.DataFrame]) -> alt.Chart
```

### Plotly Renderer (Prototype)

A lightweight Plotly backend focuses on interactive 3D scatter plots. It reuses
the shared pandas pipeline (filters + transforms) and maps the first layer of a
`PlotConfig` into `plotly.express.scatter_3d`. Current constraints:

- Single point geometry layer per plot (sufficient for demos/prototypes)
- Requires `x`, `y`, and `z` aesthetics
- Honors `color`, `size`, `opacity`, and tooltip channel lists
- Writes self-contained HTML via `Figure.write_html`

Renderer API:

```python
def build_plotly_figure(plot_cfg: PlotConfig,
                        data_registry: dict[str, pd.DataFrame]) -> plotly.Figure
```

---

## 9. Backend Capability Matrix

The following table outlines which features are supported by each backend:

| Feature           | Altair | Plotnine | Matplotlib | Plotly | Notes                                      |
| ----------------- | ------ | -------- | ---------- | ------ | ------------------------------------------ |
| **Transforms**    |        |          |            |        |                                            |
| filter            | ✅     | ✅       | ✅         | ✅     | pandas query / Vega-Lite                   |
| calculate         | ✅     | ⚠️       | ⚠️         | ✅     | Via pandas eval / limited                  |
| aggregate         | ✅     | ✅       | ✅         | ✅     | Via pandas groupby                         |
| bin               | ✅     | ✅       | ⚠️         | ✅     | Manual binning for matplotlib              |
| window            | ✅     | ⚠️       | ⚠️         | ✅     | Limited pandas support                     |
| fold/pivot        | ✅     | ✅       | ✅         | ✅     | Via pandas                                 |
| **Geometries**    |        |          |            |        |                                            |
| point, line, bar  | ✅     | ✅       | ✅         | ⚠️     | Plotly prototype currently supports point  |
| area, rect        | ✅     | ✅       | ✅         | ❌     |                                            |
| text              | ✅     | ✅       | ✅         | ❌     |                                            |
| boxplot, errorbar | ✅     | ✅       | ✅         | ❌     |                                            |
| **Interactivity** |        |          |            |        |                                            |
| Selections        | ✅     | ❌       | ❌         | ❌     | Altair-only                                |
| Tooltips          | ✅     | ❌       | ⚠️         | ✅     | Plotly hover support is built-in           |
| **Facets**        | ✅     | ✅       | ⚠️         | ⚠️     | Plotly demo does not yet facet 3D charts   |
| **Scales**        | ✅     | ✅       | ✅         | ✅     | All support log/linear/etc                 |

**Legend:**

- ✅ Fully supported
- ⚠️ Partially supported or requires workarounds
- ❌ Not supported

---

## 10. Future Backends

Because the schema is backend-agnostic, adding new rendering backends involves implementing:

```
build_plotnine_plot(plot_cfg, data_registry)
build_matplotlib_plot(plot_cfg, data_registry)
build_plotly_plot(plot_cfg, data_registry)
```

The schema remains unchanged.

---

## 11. Limitations & Future Work

- Some Vega-Lite transforms (e.g. window, fold) will need backend-specific support.
- Plotnine/matplotlib backends may not support all interactive concepts.
- More advanced layout control (grids, multi-view dashboards) is out of scope but possible.

---

## 12. Conclusion

This configuration system provides:

- A **unified Grammar-of-Graphics-style schema**
- **Declarative, reproducible plots**
- **Immediate support for Altair** (interactive plots)
- Prototype support for **Plotly 3D scatter** rendering
- A path to **plotnine/matplotlib** (static publication-ready plots)
- A clean separation between _configuration_, _data_, and _rendering backends_

By adopting this design, we gain long-term flexibility in visualization tooling while keeping plot definitions clean, expressive, and consistent across projects.

See `examples/visualization_3d_demo.py` plus the Hydra configs in
`examples/configs/visualization/` for a complete YAML-driven demo.

---

## 13. Future Extensions

### Support for other backends

- plotnine
- matplotlib

### Additional configuration

**Coordinate Systems:**
While coordinates are a core Grammar of Graphics concept, they are deferred to future work because:

- Most common use cases (cartesian coordinates) are implicit in all backends
- Polar/geographic coordinates have limited cross-backend support
- Can be added without breaking existing configs

```python
@dataclass
class CoordConfig:
    type: Literal["cartesian", "polar", "geo"]
    # Altair only supports a subset (geo; no polar without hacks).
    # plotnine supports cartesian and polar.
    # Matplotlib supports all via different projections.
```

**Theme Configuration:**
For consistent styling across plots:

```python
@dataclass
class ThemeConfig:
    # fonts, spacing, backgrounds, etc.
    font_family: str | None = None
    font_size: int | None = None
    background_color: str | None = None
    grid_color: str | None = None
    # Additional theme properties TBD
```

**Export Formats:**

- PNG, SVG, PDF for static backends
- HTML for interactive backends (Altair)
- Configurable resolution/DPI for static exports
