"""Standalone demo that renders a layered Altair chart via visualization configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    DataConfig,
    GeometryConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
    TransformConfig,
)


def main() -> None:
    """Generate a toy dataset, build a PlotConfig, and save the rendered chart."""
    df = _create_demo_dataframe()
    registry = DictDataRegistry({"metrics": df})
    plot_cfg = _build_plot_config()
    chart = build_altair_chart(plot_cfg, registry)

    output_path = Path(__file__).with_name("visualization_demo.html")
    chart.save(str(output_path))
    print(f"Wrote visualization demo to {output_path}")  # noqa: T201 - simple example harness


def _create_demo_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    records: list[dict[str, float | str | int]] = []
    for run_idx in range(3):
        run_id = f"run_{run_idx + 1}"
        for epoch in range(1, 51):
            base_loss = np.exp(-epoch / 25.0) + 0.1 * run_idx
            jitter = rng.normal(0.0, 0.02)
            loss = max(base_loss + jitter, 1e-4)
            accuracy = 0.55 + 0.008 * epoch + rng.normal(0.0, 0.01)
            records.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )
    return pd.DataFrame(records)


def _build_plot_config() -> PlotConfig:
    log_transform = TransformConfig(op="calculate", as_field="log_loss", expr="log(loss)")
    base_aesthetics = AestheticsConfig(
        x=ChannelAestheticsConfig(field="epoch", type="quantitative", title="Epoch"),
        y=ChannelAestheticsConfig(field="log_loss", type="quantitative", title="log(loss)"),
        tooltip=[
            ChannelAestheticsConfig(field="run_id", type="nominal", title="Run"),
            ChannelAestheticsConfig(field="epoch", type="quantitative", title="Epoch"),
            ChannelAestheticsConfig(field="log_loss", type="quantitative", title="log(loss)"),
        ],
    )
    raw_layer = LayerConfig(
        name="raw_runs",
        geometry=GeometryConfig(type="line", props={"opacity": 0.4}),
        aesthetics=AestheticsConfig(
            x=base_aesthetics.x,
            y=base_aesthetics.y,
            color=ChannelAestheticsConfig(field="run_id", type="nominal", title="Run"),
            tooltip=base_aesthetics.tooltip,
        ),
    )
    mean_layer = LayerConfig(
        name="mean_line",
        geometry=GeometryConfig(type="line", props={"strokeWidth": 3, "color": "#111111"}),
        aesthetics=AestheticsConfig(
            x=base_aesthetics.x,
            y=ChannelAestheticsConfig(
                field="log_loss",
                type="quantitative",
                aggregate="mean",
                title="Mean log(loss)",
            ),
        ),
    )
    return PlotConfig(
        data=DataConfig(source="metrics"),
        transforms=[log_transform],
        layers=[raw_layer, mean_layer],
        size=PlotSizeConfig(width=600, height=400),
        guides=PlotLevelGuideConfig(
            title="Training loss over epochs",
            subtitle="Each line is a synthetic training run built from random noise.",
        ),
    )


if __name__ == "__main__":
    main()
