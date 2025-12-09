"""Tests for visualization persistence helpers."""

from __future__ import annotations

import pandas as pd

from simplexity.activations.activation_visualizations import (
    ActivationVisualizationPayload,
    VisualizationControlDetail,
    VisualizationControlsState,
    render_visualization,
)
from simplexity.activations.visualization_persistence import save_visualization_payloads
from simplexity.visualization.history import history_paths
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    DataConfig,
    GeometryConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)


def _plot_config() -> PlotConfig:
    layer = LayerConfig(
        geometry=GeometryConfig(type="line"),
        aesthetics=AestheticsConfig(
            x=ChannelAestheticsConfig(field="step", type="quantitative"),
            y=ChannelAestheticsConfig(field="value", type="quantitative"),
        ),
    )
    return PlotConfig(
        backend="altair",
        data=DataConfig(source="main"),
        layers=[layer],
        size=PlotSizeConfig(),
        guides=PlotLevelGuideConfig(),
    )


def _payload(dataframe: pd.DataFrame) -> ActivationVisualizationPayload:
    cfg = _plot_config()
    controls = VisualizationControlsState(
        slider=VisualizationControlDetail(
            type="slider",
            field="step",
            options=list(pd.unique(dataframe["step"])) if "step" in dataframe else [],
        )
    )
    figure = render_visualization(cfg, dataframe, controls)
    return ActivationVisualizationPayload(
        analysis="analysis",
        name="viz",
        backend="altair",
        figure=figure,
        dataframe=dataframe,
        controls=controls,
        plot_config=cfg,
    )


def test_save_visualization_payloads_accumulates_step_history(tmp_path):
    """Test that visualization payloads accumulate history across steps."""
    df_first = pd.DataFrame({"step": [0, 0], "value": [0.1, 0.2]})
    payload_one = _payload(df_first)

    save_visualization_payloads({"analysis/viz": payload_one}, tmp_path, step=1)

    data_path, _ = history_paths(tmp_path, "analysis_viz")
    assert data_path.exists()
    history_df = pd.read_json(data_path, orient="records", lines=True)
    assert len(history_df) == len(df_first)
    assert set(history_df["step"]) == {1}
    assert set(history_df["sequence_step"]) == {0}
    assert (tmp_path / "step_00001" / "analysis" / "viz.html").exists()

    df_second = pd.DataFrame({"step": [1], "value": [0.5]})
    payload_two = _payload(df_second)

    save_visualization_payloads({"analysis/viz": payload_two}, tmp_path, step=2)

    history_df = pd.read_json(data_path, orient="records", lines=True)
    assert len(history_df) == len(df_first) + len(df_second)
    assert set(history_df["step"]) == {1, 2}
    assert set(history_df["sequence_step"]) == {0, 1}
    assert (tmp_path / "step_00002" / "analysis" / "viz.html").exists()
