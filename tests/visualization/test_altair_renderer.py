"""Tests for altair renderer."""

import pandas as pd
import pytest

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.altair_renderer import (
    _apply_geometry,
    _build_layer_chart,
    _encode_aesthetics,
    build_altair_chart,
)
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    ChannelAestheticsConfig,
    DataConfig,
    FacetConfig,
    GeometryConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)

try:
    import altair as alt
except ImportError:
    pytest.skip("Altair not installed", allow_module_level=True)


class TestBuildAltairChart:
    """Tests for build_altair_chart function."""

    def test_raises_when_no_layers(self):
        """Test that empty layers raises error."""
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[])
        registry = DictDataRegistry({"main": pd.DataFrame()})
        with pytest.raises(ConfigValidationError, match="at least one layer"):
            build_altair_chart(plot_cfg, registry)

    def test_builds_simple_point_chart(self):
        """Test building a simple point chart."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_builds_line_chart(self):
        """Test building a line chart."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="line"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_builds_bar_chart(self):
        """Test building a bar chart."""
        df = pd.DataFrame({"category": ["a", "b", "c"], "value": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="bar"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="category", type="nominal"),
                y=ChannelAestheticsConfig(field="value", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_applies_color_encoding(self):
        """Test that color encoding is applied."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "cat": ["a", "b", "a"]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                color=ChannelAestheticsConfig(field="cat", type="nominal"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_applies_size(self):
        """Test that chart size is applied."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        size = PlotSizeConfig(width=800, height=600)
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], size=size)
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_applies_guides(self):
        """Test that plot guides are applied."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        guides = PlotLevelGuideConfig(title="My Chart")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], guides=guides)
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_multiple_layers(self):
        """Test building chart with multiple layers."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer1 = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        layer2 = LayerConfig(
            geometry=GeometryConfig(type="line"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer1, layer2])
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None


class TestApplyGeometry:
    """Tests for _apply_geometry function."""

    def test_point_geometry(self):
        """Test point geometry application."""
        chart = alt.Chart(pd.DataFrame({"x": [1]}))
        geometry = GeometryConfig(type="point")
        result = _apply_geometry(chart, geometry)
        assert result is not None

    def test_line_geometry(self):
        """Test line geometry application."""
        chart = alt.Chart(pd.DataFrame({"x": [1]}))
        geometry = GeometryConfig(type="line")
        result = _apply_geometry(chart, geometry)
        assert result is not None

    def test_bar_geometry(self):
        """Test bar geometry application."""
        chart = alt.Chart(pd.DataFrame({"x": [1]}))
        geometry = GeometryConfig(type="bar")
        result = _apply_geometry(chart, geometry)
        assert result is not None

    def test_area_geometry(self):
        """Test area geometry application."""
        chart = alt.Chart(pd.DataFrame({"x": [1]}))
        geometry = GeometryConfig(type="area")
        result = _apply_geometry(chart, geometry)
        assert result is not None

    def test_invalid_geometry_raises(self):
        """Test that invalid geometry type raises error."""
        chart = alt.Chart(pd.DataFrame({"x": [1]}))
        geometry = GeometryConfig(type="invalid_type")
        with pytest.raises(ConfigValidationError, match="does not support geometry"):
            _apply_geometry(chart, geometry)


class TestEncodeAesthetics:
    """Tests for _encode_aesthetics function."""

    def test_basic_x_y_encoding(self):
        """Test basic x and y encoding."""
        aes = AestheticsConfig(
            x=ChannelAestheticsConfig(field="x", type="quantitative"),
            y=ChannelAestheticsConfig(field="y", type="quantitative"),
        )
        encoding = _encode_aesthetics(aes)
        assert "x" in encoding
        assert "y" in encoding

    def test_color_encoding(self):
        """Test color encoding."""
        aes = AestheticsConfig(
            x=ChannelAestheticsConfig(field="x", type="quantitative"),
            color=ChannelAestheticsConfig(field="cat", type="nominal"),
        )
        encoding = _encode_aesthetics(aes)
        assert "color" in encoding

    def test_size_encoding(self):
        """Test size encoding."""
        aes = AestheticsConfig(
            x=ChannelAestheticsConfig(field="x", type="quantitative"),
            size=ChannelAestheticsConfig(field="size", type="quantitative"),
        )
        encoding = _encode_aesthetics(aes)
        assert "size" in encoding

    def test_opacity_encoding(self):
        """Test opacity encoding."""
        aes = AestheticsConfig(
            x=ChannelAestheticsConfig(field="x", type="quantitative"),
            opacity=ChannelAestheticsConfig(field="opacity", type="quantitative"),
        )
        encoding = _encode_aesthetics(aes)
        assert "opacity" in encoding

    def test_empty_aesthetics_returns_empty(self):
        """Test that empty aesthetics returns empty dict."""
        aes = AestheticsConfig()
        encoding = _encode_aesthetics(aes)
        assert not encoding


class TestFaceting:
    """Tests for faceted charts."""

    def test_column_facet(self):
        """Test column faceting."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "group": ["a", "a", "b", "b"]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        facet = FacetConfig(column="group")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_row_facet(self):
        """Test row faceting."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "group": ["a", "a", "b", "b"]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        facet = FacetConfig(row="group")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None

    def test_row_and_column_facet(self):
        """Test both row and column faceting."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8],
                "y": [4, 5, 6, 7, 8, 9, 10, 11],
                "row_group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "col_group": ["x", "x", "y", "y", "x", "x", "y", "y"],
            }
        )
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        facet = FacetConfig(row="row_group", column="col_group")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        chart = build_altair_chart(plot_cfg, registry)
        assert chart is not None


class TestBuildLayerChart:
    """Tests for _build_layer_chart function."""

    def test_builds_chart_from_layer(self):
        """Test building a chart from layer config."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        chart = _build_layer_chart(layer, df)
        assert chart is not None

    def test_applies_geometry_props(self):
        """Test that geometry props are applied."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point", props={"size": 100}),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        chart = _build_layer_chart(layer, df)
        assert chart is not None
