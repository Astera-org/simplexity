"""Demo script that renders charts with different theme configurations.

This script creates the same chart multiple times with different visual themes,
outputting HTML files that can be viewed in a browser to compare themes.

Run with:
    uv run python tests/end_to_end/theme_demo.py

Outputs:
    tests/end_to_end/theme_outputs/theme_light.html
    tests/end_to_end/theme_outputs/theme_dark.html
    tests/end_to_end/theme_outputs/theme_minimal.html
    tests/end_to_end/theme_outputs/theme_publication.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    AxisConfig,
    ChannelAestheticsConfig,
    DataConfig,
    GeometryConfig,
    LayerConfig,
    LegendConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
    ScaleConfig,
)


@dataclass
class ThemeDefinition:
    """Temporary theme definition until ThemeConfig is added to structured_configs."""

    name: str
    background_color: str
    font_family: str
    font_size: int
    title_font_size: int
    axis_color: str
    grid_color: str | None
    text_color: str


THEMES: dict[str, ThemeDefinition] = {
    "light": ThemeDefinition(
        name="light",
        background_color="#ffffff",
        font_family="Arial, sans-serif",
        font_size=12,
        title_font_size=16,
        axis_color="#333333",
        grid_color="#e0e0e0",
        text_color="#333333",
    ),
    "dark": ThemeDefinition(
        name="dark",
        background_color="#1e1e1e",
        font_family="Arial, sans-serif",
        font_size=12,
        title_font_size=16,
        axis_color="#cccccc",
        grid_color="#444444",
        text_color="#ffffff",
    ),
    "minimal": ThemeDefinition(
        name="minimal",
        background_color="#ffffff",
        font_family="Helvetica, sans-serif",
        font_size=11,
        title_font_size=14,
        axis_color="#666666",
        grid_color=None,
        text_color="#333333",
    ),
    "publication": ThemeDefinition(
        name="publication",
        background_color="#ffffff",
        font_family="Times New Roman, serif",
        font_size=10,
        title_font_size=12,
        axis_color="#000000",
        grid_color="#cccccc",
        text_color="#000000",
    ),
}


def main() -> None:
    """Generate themed charts and save them to HTML files."""
    output_dir = Path(__file__).parent / "theme_outputs"
    output_dir.mkdir(exist_ok=True)

    df = _create_demo_dataframe()
    registry = DictDataRegistry({"main": df})

    for theme_name, theme in THEMES.items():
        _render_altair_themed(registry, theme, output_dir)
        _render_plotly_themed(registry, theme, output_dir)

    print(f"Wrote themed visualizations to {output_dir}/")  # noqa: T201
    print("Open the HTML files in a browser to compare themes.")  # noqa: T201


def _create_demo_dataframe() -> pd.DataFrame:
    """Create sample data for visualization."""
    rng = np.random.default_rng(42)
    n_points = 100

    x = np.linspace(0, 10, n_points)
    categories = rng.choice(["Group A", "Group B", "Group C"], size=n_points)

    records = []
    for i, (xi, cat) in enumerate(zip(x, categories, strict=True)):
        base_y = np.sin(xi) + {"Group A": 0, "Group B": 1, "Group C": 2}[cat]
        y = base_y + rng.normal(0, 0.2)
        records.append({"x": xi, "y": y, "category": cat, "size": rng.uniform(5, 20)})

    return pd.DataFrame(records)


def _build_base_plot_config(backend: str) -> PlotConfig:
    """Build a base plot config that can be themed."""
    layer = LayerConfig(
        geometry=GeometryConfig(type="point", props={"size": 60} if backend == "altair" else {}),
        aesthetics=AestheticsConfig(
            x=ChannelAestheticsConfig(
                field="x",
                type="quantitative",
                title="X Value",
                axis=AxisConfig(grid=True),
            ),
            y=ChannelAestheticsConfig(
                field="y",
                type="quantitative",
                title="Y Value",
                axis=AxisConfig(grid=True),
            ),
            color=ChannelAestheticsConfig(
                field="category",
                type="nominal",
                title="Category",
                legend=LegendConfig(title="Groups"),
            ),
            size=ChannelAestheticsConfig(field="size", type="quantitative")
            if backend == "plotly"
            else None,
        ),
    )

    return PlotConfig(
        backend=backend,
        data=DataConfig(source="main"),
        layers=[layer],
        size=PlotSizeConfig(width=600, height=400),
        guides=PlotLevelGuideConfig(
            title="Theme Demo Chart",
            subtitle="Comparing visual themes",
        ),
    )


def _render_altair_themed(
    registry: DictDataRegistry, theme: ThemeDefinition, output_dir: Path
) -> None:
    """Render an Altair chart with theme applied via configure_* methods."""
    try:
        import altair as alt
    except ImportError:
        print("Altair not installed, skipping Altair themes")  # noqa: T201
        return

    plot_cfg = _build_base_plot_config(backend="altair")
    plot_cfg.background = theme.background_color

    chart = build_altair_chart(plot_cfg, registry)

    themed_chart = (
        chart.configure(background=theme.background_color)
        .configure_axis(
            labelColor=theme.text_color,
            titleColor=theme.text_color,
            gridColor=theme.grid_color if theme.grid_color else "transparent",
            domainColor=theme.axis_color,
            tickColor=theme.axis_color,
            labelFont=theme.font_family,
            titleFont=theme.font_family,
            labelFontSize=theme.font_size,
            titleFontSize=theme.font_size,
        )
        .configure_legend(
            labelColor=theme.text_color,
            titleColor=theme.text_color,
            labelFont=theme.font_family,
            titleFont=theme.font_family,
            labelFontSize=theme.font_size,
            titleFontSize=theme.font_size,
        )
        .configure_title(
            color=theme.text_color,
            font=theme.font_family,
            fontSize=theme.title_font_size,
            subtitleColor=theme.text_color,
            subtitleFont=theme.font_family,
            subtitleFontSize=theme.font_size,
        )
        .configure_view(strokeWidth=0)
    )

    output_path = output_dir / f"altair_theme_{theme.name}.html"
    themed_chart.save(str(output_path))
    print(f"  Saved: {output_path.name}")  # noqa: T201


def _render_plotly_themed(
    registry: DictDataRegistry, theme: ThemeDefinition, output_dir: Path
) -> None:
    """Render a Plotly chart with theme applied via update_layout."""
    plot_cfg = _build_base_plot_config(backend="plotly")
    plot_cfg.background = theme.background_color

    # Add z for 3D (required by plotly renderer)
    plot_cfg.layers[0].aesthetics.z = ChannelAestheticsConfig(
        field="y", type="quantitative", title="Z Value"
    )

    fig = build_plotly_figure(plot_cfg, registry)

    font_config: dict[str, Any] = {
        "family": theme.font_family,
        "size": theme.font_size,
        "color": theme.text_color,
    }

    fig.update_layout(
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        font=font_config,
        title_font={"size": theme.title_font_size, "color": theme.text_color},
        scene={
            "xaxis": {
                "backgroundcolor": theme.background_color,
                "gridcolor": theme.grid_color or "rgba(0,0,0,0)",
                "color": theme.text_color,
                "showgrid": theme.grid_color is not None,
            },
            "yaxis": {
                "backgroundcolor": theme.background_color,
                "gridcolor": theme.grid_color or "rgba(0,0,0,0)",
                "color": theme.text_color,
                "showgrid": theme.grid_color is not None,
            },
            "zaxis": {
                "backgroundcolor": theme.background_color,
                "gridcolor": theme.grid_color or "rgba(0,0,0,0)",
                "color": theme.text_color,
                "showgrid": theme.grid_color is not None,
            },
        },
        legend={
            "font": font_config,
            "title": {"font": font_config},
        },
    )

    output_path = output_dir / f"plotly_theme_{theme.name}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Saved: {output_path.name}")  # noqa: T201


if __name__ == "__main__":
    main()
