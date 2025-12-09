"""Tests for renderer support of interactive controls."""

from __future__ import annotations

import pandas as pd

from simplexity.activations.activation_visualizations import (
    VisualizationControlDetail,
    VisualizationControlsState,
)
from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import build_plotly_figure
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


def _base_plot_config(backend: str = "altair") -> PlotConfig:
    layer = LayerConfig(
        geometry=GeometryConfig(type="line" if backend == "altair" else "point"),
        aesthetics=AestheticsConfig(
            x=ChannelAestheticsConfig(field="x", type="quantitative"),
            y=ChannelAestheticsConfig(field="y", type="quantitative"),
            z=ChannelAestheticsConfig(field="z", type="quantitative") if backend == "plotly" else None,
        ),
    )
    return PlotConfig(
        backend=backend,
        data=DataConfig(source="main"),
        layers=[layer],
        size=PlotSizeConfig(),
        guides=PlotLevelGuideConfig(),
    )


def _layer_controls(values: list[str]) -> VisualizationControlsState:
    dropdown = VisualizationControlDetail(type="dropdown", field="layer", options=values)
    return VisualizationControlsState(dropdown=dropdown)


def _slider_controls(values: list[int]) -> VisualizationControlsState:
    slider = VisualizationControlDetail(type="slider", field="step", options=values)
    return VisualizationControlsState(slider=slider)


def test_altair_renderer_adds_dropdown_selection():
    """Altair renderer should add a dropdown param when controls include layer dropdown."""
    df = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 1, 1, 0],
            "layer": ["layer_0", "layer_0", "layer_1", "layer_1"],
        }
    )
    plot_cfg = _base_plot_config(backend="altair")
    registry = DictDataRegistry({"main": df})
    controls = _layer_controls(["layer_0", "layer_1"])

    chart = build_altair_chart(plot_cfg, registry, controls=controls)
    spec = chart.to_dict()

    assert "params" in spec
    assert spec["params"][0]["name"] == "layer_dropdown"
    assert spec["params"][0]["bind"]["options"] == ["layer_0", "layer_1"]


def test_altair_renderer_adds_slider_binding():
    df = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 1, 1, 0],
            "step": [0, 0, 1, 1],
        }
    )
    plot_cfg = _base_plot_config(backend="altair")
    registry = DictDataRegistry({"main": df})
    controls = _slider_controls([0, 1])

    chart = build_altair_chart(plot_cfg, registry, controls=controls)
    spec = chart.to_dict()

    assert any(param["name"].endswith("_slider") for param in spec.get("params", []))
    slider_param = next(param for param in spec["params"] if param["name"].endswith("_slider"))
    assert slider_param["bind"]["input"] in {"range", "select"}


def test_altair_renderer_skips_slider_when_accumulating():
    df = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 1, 1, 0],
            "step": [0, 0, 1, 1],
        }
    )
    plot_cfg = _base_plot_config(backend="altair")
    registry = DictDataRegistry({"main": df})
    controls = VisualizationControlsState(
        slider=VisualizationControlDetail(type="slider", field="step", options=[0, 1]),
        accumulate_steps=True,
    )

    chart = build_altair_chart(plot_cfg, registry, controls=controls)
    spec = chart.to_dict()

    assert all(not param["name"].endswith("_slider") for param in spec.get("params", []))


def test_altair_renderer_injects_detail_when_accumulating():
    df = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 1, 1, 0],
            "step": [0, 0, 1, 1],
        }
    )
    plot_cfg = _base_plot_config(backend="altair")
    registry = DictDataRegistry({"main": df})
    controls = VisualizationControlsState(accumulate_steps=True)

    chart = build_altair_chart(plot_cfg, registry, controls=controls)
    spec = chart.to_dict()

    assert "detail" in spec.get("encoding", {})
    detail_encoding = spec["encoding"]["detail"]
    if isinstance(detail_encoding, list):
        detail_encoding = detail_encoding[0]
    assert detail_encoding["field"] == "step"


def test_altair_renderer_skips_detail_when_step_axis_used():
    df = pd.DataFrame(
        {
            "step": [0, 1, 2, 3],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )
    plot_cfg = _base_plot_config(backend="altair")
    plot_cfg.layers[0].aesthetics.x.field = "step"
    plot_cfg.layers[0].aesthetics.y.field = "y"
    registry = DictDataRegistry({"main": df})
    controls = VisualizationControlsState(accumulate_steps=True)

    chart = build_altair_chart(plot_cfg, registry, controls=controls)
    spec = chart.to_dict()

    assert "detail" not in spec.get("encoding", {})


def test_plotly_renderer_adds_layer_dropdown_menu():
    """Plotly renderer should add a dropdown menu that toggles layer visibility."""
    df = pd.DataFrame(
        {
            "layer": ["layer_0"] * 5 + ["layer_1"] * 5,
            "x": list(range(10)),
            "y": [value * 0.5 for value in range(10)],
            "z": [1.0] * 10,
        }
    )
    plot_cfg = _base_plot_config(backend="plotly")
    registry = DictDataRegistry({"main": df})
    controls = _layer_controls(["layer_0", "layer_1"])

    figure = build_plotly_figure(plot_cfg, registry, controls=controls)

    assert figure.layout.updatemenus
    menu = figure.layout.updatemenus[0]
    assert len(menu["buttons"]) == 2
    assert [button["label"] for button in menu["buttons"]] == ["layer_0", "layer_1"]
    # First trace should be visible initially, remaining traces hidden until selected.
    assert figure.data[0].visible is True
    assert all(trace.visible is False for trace in figure.data[1:])


def test_plotly_renderer_adds_step_slider():
    df = pd.DataFrame(
        {
            "layer": ["layer_0"] * 6 + ["layer_1"] * 6,
            "x": list(range(12)),
            "y": [value * 0.5 for value in range(12)],
            "z": [1.0] * 12,
            "step": [0, 0, 1, 1, 2, 2] * 2,
        }
    )
    plot_cfg = _base_plot_config(backend="plotly")
    registry = DictDataRegistry({"main": df})
    controls = VisualizationControlsState(
        dropdown=VisualizationControlDetail(type="dropdown", field="layer", options=["layer_0", "layer_1"]),
        slider=VisualizationControlDetail(type="slider", field="step", options=[0, 1, 2]),
    )

    figure = build_plotly_figure(plot_cfg, registry, controls=controls)

    assert figure.layout.sliders
    assert len(figure.frames) == 3


def test_plotly_renderer_preserves_literal_colors():
    df = pd.DataFrame(
        {
            "x": [0, 1],
            "y": [0, 1],
            "z": [0, 1],
            "literal_color": ["#00ff00", "#ff0000"],
        }
    )
    plot_cfg = _base_plot_config(backend="plotly")
    plot_cfg.layers[0].aesthetics.color = ChannelAestheticsConfig(field="literal_color", type="nominal")
    registry = DictDataRegistry({"main": df})

    figure = build_plotly_figure(plot_cfg, registry)

    assert figure.data
    assert list(figure.data[0].marker.color) == ["#00ff00", "#ff0000"]
