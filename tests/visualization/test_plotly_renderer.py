"""Tests for plotly renderer."""

import pandas as pd
import pytest

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import (
    _axis_title,
    _build_scatter2d,
    _build_scatter3d,
    _require_field,
    _resolve_layer_dropdown,
    _resolve_slider_control,
    build_plotly_figure,
)
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


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_axis_title_from_config(self):
        """Test axis title extraction from config."""
        config = ChannelAestheticsConfig(field="x", type="quantitative", title="X Axis")
        assert _axis_title(config) == "X Axis"

    def test_axis_title_none_when_no_config(self):
        """Test axis title is None when no config."""
        assert _axis_title(None) is None

    def test_axis_title_uses_field_when_no_title(self):
        """Test axis title falls back to field name when no title."""
        config = ChannelAestheticsConfig(field="x", type="quantitative")
        assert _axis_title(config) == "x"

    def test_require_field_extracts_field(self):
        """Test that require_field extracts the field name."""
        config = ChannelAestheticsConfig(field="my_field", type="quantitative")
        assert _require_field(config, "x") == "my_field"

    def test_require_field_raises_when_none(self):
        """Test that require_field raises when config is None."""
        with pytest.raises(ConfigValidationError, match="requires"):
            _require_field(None, "x")

    def test_require_field_raises_when_no_field(self):
        """Test that require_field raises when field is None."""
        config = ChannelAestheticsConfig(field=None, type="quantitative")
        with pytest.raises(ConfigValidationError, match="requires"):
            _require_field(config, "x")


class TestResolveControls:
    """Tests for control resolution functions."""

    def test_resolve_slider_control_none_when_no_controls(self):
        """Test slider returns None when no controls."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _resolve_slider_control(df, None)
        assert result is None

    def test_resolve_layer_dropdown_none_when_no_controls(self):
        """Test layer dropdown returns None when no controls."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _resolve_layer_dropdown(df, None)
        assert result is None


class TestBuildScatter2D:
    """Tests for 2D scatter plot building."""

    def test_basic_scatter2d(self):
        """Test basic 2D scatter plot building."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        fig = _build_scatter2d(layer, df, None)
        assert fig is not None
        assert len(fig.data) > 0

    def test_scatter2d_with_color(self):
        """Test 2D scatter with color encoding."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "category": ["a", "b", "a"]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                color=ChannelAestheticsConfig(field="category", type="nominal"),
            ),
        )
        fig = _build_scatter2d(layer, df, None)
        assert fig is not None


class TestBuildScatter3D:
    """Tests for 3D scatter plot building."""

    def test_basic_scatter3d(self):
        """Test basic 3D scatter plot building."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                z=ChannelAestheticsConfig(field="z", type="quantitative"),
            ),
        )
        fig = _build_scatter3d(layer, df, None)
        assert fig is not None
        assert len(fig.data) > 0

    def test_scatter3d_with_color(self):
        """Test 3D scatter with color encoding."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9], "cat": ["a", "b", "a"]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                z=ChannelAestheticsConfig(field="z", type="quantitative"),
                color=ChannelAestheticsConfig(field="cat", type="nominal"),
            ),
        )
        fig = _build_scatter3d(layer, df, None)
        assert fig is not None


class TestBuildPlotlyFigure:
    """Tests for the main build_plotly_figure function."""

    def test_raises_when_no_layers(self):
        """Test that empty layers raises error."""
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[])
        registry = DictDataRegistry({"main": pd.DataFrame()})
        with pytest.raises(ConfigValidationError, match="at least one layer"):
            build_plotly_figure(plot_cfg, registry)

    def test_raises_when_multiple_layers(self):
        """Test that multiple layers raises error (currently unsupported)."""
        layer1 = LayerConfig(geometry=GeometryConfig(type="point"))
        layer2 = LayerConfig(geometry=GeometryConfig(type="point"))
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer1, layer2])
        registry = DictDataRegistry({"main": pd.DataFrame({"x": [1], "y": [2]})})
        with pytest.raises(ConfigValidationError, match="exactly one layer"):
            build_plotly_figure(plot_cfg, registry)

    def test_raises_when_non_point_geometry(self):
        """Test that non-point geometry raises error."""
        layer = LayerConfig(geometry=GeometryConfig(type="line"))
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": pd.DataFrame({"x": [1], "y": [2]})})
        with pytest.raises(ConfigValidationError, match="point geometry"):
            build_plotly_figure(plot_cfg, registry)

    def test_builds_2d_figure(self):
        """Test building a basic 2D figure."""
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
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None

    def test_builds_3d_figure(self):
        """Test building a basic 3D figure."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                z=ChannelAestheticsConfig(field="z", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer])
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None

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
        guides = PlotLevelGuideConfig(title="My Plot", subtitle="My Subtitle")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], guides=guides)
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert "My Plot" in fig.layout.title.text

    def test_applies_size(self):
        """Test that plot size is applied."""
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
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig.layout.width == 800
        assert fig.layout.height == 600


class TestFacetedFigures:
    """Tests for faceted figure building."""

    def test_builds_column_faceted_figure(self):
        """Test building a column-faceted 2D figure."""
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
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None

    def test_builds_row_faceted_figure(self):
        """Test building a row-faceted 2D figure."""
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
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None

    def test_builds_3d_faceted_figure(self):
        """Test building a 3D faceted figure."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [4, 5, 6, 7],
                "z": [7, 8, 9, 10],
                "group": ["a", "a", "b", "b"],
            }
        )
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                z=ChannelAestheticsConfig(field="z", type="quantitative"),
            ),
        )
        facet = FacetConfig(column="group")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None

    def test_builds_row_and_column_faceted_figure(self):
        """Test building figure with both row and column facets."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6, 7, 8],
                "y": [1, 2, 3, 4, 5, 6, 7, 8],
                "row_grp": ["r1", "r1", "r1", "r1", "r2", "r2", "r2", "r2"],
                "col_grp": ["c1", "c1", "c2", "c2", "c1", "c1", "c2", "c2"],
            }
        )
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        facet = FacetConfig(row="row_grp", column="col_grp")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None


class TestScatterWithEncodings:
    """Tests for scatter plots with various encodings."""

    def test_scatter2d_with_size_encoding(self):
        """Test 2D scatter with size encoding."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "size_val": [10, 20, 30]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                size=ChannelAestheticsConfig(field="size_val", type="quantitative"),
            ),
        )
        fig = _build_scatter2d(layer, df, None)
        assert fig is not None

    def test_scatter2d_with_opacity(self):
        """Test 2D scatter with opacity encoding."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                opacity=ChannelAestheticsConfig(field=None, type="quantitative", value=0.5),
            ),
        )
        fig = _build_scatter2d(layer, df, None)
        assert fig is not None

    def test_scatter3d_with_size_encoding(self):
        """Test 3D scatter with size encoding."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9], "size_val": [10, 20, 30]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                z=ChannelAestheticsConfig(field="z", type="quantitative"),
                size=ChannelAestheticsConfig(field="size_val", type="quantitative"),
            ),
        )
        fig = _build_scatter3d(layer, df, None)
        assert fig is not None

    def test_figure_with_background_color(self):
        """Test that background color is applied."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
            ),
        )
        plot_cfg = PlotConfig(
            data=DataConfig(source="main"),
            layers=[layer],
            background="#f0f0f0",
        )
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig.layout.plot_bgcolor == "#f0f0f0"

    def test_faceted_figure_with_color_encoding(self):
        """Test faceted figure with color encoding."""
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [4, 5, 6, 7],
                "group": ["a", "a", "b", "b"],
                "category": ["cat1", "cat2", "cat1", "cat2"],
            }
        )
        layer = LayerConfig(
            geometry=GeometryConfig(type="point"),
            aesthetics=AestheticsConfig(
                x=ChannelAestheticsConfig(field="x", type="quantitative"),
                y=ChannelAestheticsConfig(field="y", type="quantitative"),
                color=ChannelAestheticsConfig(field="category", type="nominal"),
            ),
        )
        facet = FacetConfig(column="group")
        plot_cfg = PlotConfig(data=DataConfig(source="main"), layers=[layer], facet=facet)
        registry = DictDataRegistry({"main": df})
        fig = build_plotly_figure(plot_cfg, registry)
        assert fig is not None
