"""Tests for data pipeline transforms and materialization."""

import pandas as pd
import pytest

from simplexity.exceptions import ConfigValidationError
from simplexity.visualization.data_pipeline import (
    _apply_transform,
    _derive_fold_names,
    _parse_function_expr,
    apply_filters,
    apply_transforms,
    build_plot_level_dataframe,
    materialize_data,
    normalize_expression,
    resolve_layer_dataframe,
)
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.structured_configs import (
    DataConfig,
    LayerConfig,
    TransformConfig,
)


class TestNormalizeExpression:
    """Tests for normalize_expression."""

    def test_removes_datum_prefix(self):
        """Test that datum. prefix is removed."""
        assert normalize_expression("datum.x > 5") == "x > 5"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert normalize_expression("  x > 5  ") == "x > 5"


class TestApplyFilters:
    """Tests for apply_filters."""

    def test_single_filter(self):
        """Test applying a single filter."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        result = apply_filters(df, ["x > 2"])
        assert list(result["x"]) == [3, 4, 5]

    def test_multiple_filters(self):
        """Test applying multiple filters."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        result = apply_filters(df, ["x > 2", "y < 50"])
        assert list(result["x"]) == [3, 4]

    def test_filter_with_datum_prefix(self):
        """Test that datum. prefix is normalized."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_filters(df, ["datum.x > 1"])
        assert list(result["x"]) == [2, 3]


class TestMaterializeData:
    """Tests for materialize_data."""

    def test_basic_materialization(self):
        """Test basic data materialization."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main")
        result = materialize_data(data_cfg, registry)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3

    def test_with_filters(self):
        """Test materialization with filters."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main", filters=["x > 1"])
        result = materialize_data(data_cfg, registry)
        assert len(result) == 2

    def test_with_column_selection(self):
        """Test materialization with column selection."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main", columns=["x", "z"])
        result = materialize_data(data_cfg, registry)
        assert list(result.columns) == ["x", "z"]

    def test_missing_column_raises(self):
        """Test that missing columns raise an error."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main", columns=["x", "missing"])
        with pytest.raises(ConfigValidationError, match="not present"):
            materialize_data(data_cfg, registry)


class TestBuildPlotLevelDataframe:
    """Tests for build_plot_level_dataframe."""

    def test_with_transforms(self):
        """Test building dataframe with transforms."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main")
        transforms = [TransformConfig(op="calculate", expr="x * 2", as_field="x2")]
        result = build_plot_level_dataframe(data_cfg, transforms, registry)
        assert "x2" in result.columns
        assert list(result["x2"]) == [2, 4, 6]

    def test_without_transforms(self):
        """Test building dataframe without transforms."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        registry = DictDataRegistry({"main": df})
        data_cfg = DataConfig(source="main")
        result = build_plot_level_dataframe(data_cfg, [], registry)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3


class TestResolveLayerDataframe:
    """Tests for resolve_layer_dataframe."""

    def test_uses_plot_df_when_no_layer_data(self):
        """Test that plot dataframe is used when layer has no data config."""
        plot_df = pd.DataFrame({"x": [1, 2, 3]})
        layer = LayerConfig()
        result = resolve_layer_dataframe(layer, plot_df, {})
        assert list(result["x"]) == [1, 2, 3]

    def test_uses_layer_data_when_specified(self):
        """Test that layer data config is used when specified."""
        plot_df = pd.DataFrame({"x": [1, 2, 3]})
        layer_df = pd.DataFrame({"y": [4, 5, 6]})
        registry = DictDataRegistry({"layer_data": layer_df})
        layer = LayerConfig(data=DataConfig(source="layer_data"))
        result = resolve_layer_dataframe(layer, plot_df, registry)
        assert "y" in result.columns
        assert "x" not in result.columns

    def test_applies_layer_transforms(self):
        """Test that layer transforms are applied."""
        plot_df = pd.DataFrame({"x": [1, 2, 3]})
        layer = LayerConfig(transforms=[TransformConfig(op="filter", filter="x > 1")])
        result = resolve_layer_dataframe(layer, plot_df, {})
        assert len(result) == 2


class TestApplyTransform:
    """Tests for individual transform operations."""

    def test_filter_transform(self):
        """Test filter transform."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="filter", filter="x > 1")
        result = _apply_transform(df, transform)
        assert len(result) == 2

    def test_filter_requires_expression(self):
        """Test that filter transform requires filter expression."""
        with pytest.raises(ConfigValidationError, match="filter"):
            TransformConfig(op="filter")

    def test_calculate_transform(self):
        """Test calculate transform."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="calculate", expr="x * 2", as_field="x2")
        result = _apply_transform(df, transform)
        assert list(result["x2"]) == [2, 4, 6]

    def test_calculate_requires_as_field(self):
        """Test that calculate transform requires as_field."""
        with pytest.raises(ConfigValidationError, match="as_field"):
            TransformConfig(op="calculate", expr="x * 2")

    def test_aggregate_transform(self):
        """Test aggregate transform."""
        df = pd.DataFrame({"group": ["a", "a", "b"], "value": [1, 2, 3]})
        transform = TransformConfig(op="aggregate", groupby=["group"], aggregations={"total": "sum(value)"})
        result = _apply_transform(df, transform)
        assert len(result) == 2
        assert "total" in result.columns

    def test_aggregate_requires_groupby_and_aggregations(self):
        """Test that aggregate transform requires groupby and aggregations."""
        with pytest.raises(ConfigValidationError, match="groupby"):
            TransformConfig(op="aggregate")

    def test_bin_transform(self):
        """Test bin transform."""
        df = pd.DataFrame({"x": [1, 5, 10, 15, 20]})
        transform = TransformConfig(op="bin", field="x", binned_as="x_bin", maxbins=5)
        result = _apply_transform(df, transform)
        assert "x_bin" in result.columns

    def test_bin_requires_field_and_binned_as(self):
        """Test that bin transform requires field and binned_as."""
        with pytest.raises(ConfigValidationError, match="field"):
            TransformConfig(op="bin")

    def test_window_transform_rank(self):
        """Test window transform with rank function."""
        df = pd.DataFrame({"x": [3, 1, 2]})
        transform = TransformConfig(op="window", window={"x_rank": "rank(x)"})
        result = _apply_transform(df, transform)
        assert "x_rank" in result.columns

    def test_window_transform_cumsum(self):
        """Test window transform with cumsum function."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="window", window={"x_cumsum": "cumsum(x)"})
        result = _apply_transform(df, transform)
        assert list(result["x_cumsum"]) == [1, 3, 6]

    def test_window_unsupported_function(self):
        """Test that unsupported window function raises error."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="window", window={"x_bad": "unsupported(x)"})
        with pytest.raises(ConfigValidationError, match="not supported"):
            _apply_transform(df, transform)

    def test_window_requires_window_mapping(self):
        """Test that window transform requires window mapping."""
        with pytest.raises(ConfigValidationError, match="window"):
            TransformConfig(op="window")

    def test_fold_transform(self):
        """Test fold transform."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        transform = TransformConfig(op="fold", fold_fields=["a", "b"])
        result = _apply_transform(df, transform)
        assert "key" in result.columns
        assert "value" in result.columns
        assert len(result) == 4

    def test_fold_requires_fold_fields(self):
        """Test that fold transform requires fold_fields."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="fold")
        with pytest.raises(ConfigValidationError, match="fold_fields"):
            _apply_transform(df, transform)

    def test_pivot_not_implemented(self):
        """Test that pivot transform is not implemented."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="pivot")
        with pytest.raises(ConfigValidationError, match="not implemented"):
            _apply_transform(df, transform)

    def test_unsupported_op_raises(self):
        """Test that unsupported operation raises error."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        transform = TransformConfig(op="unknown")
        with pytest.raises(ConfigValidationError, match="Unsupported"):
            _apply_transform(df, transform)


class TestApplyTransforms:
    """Tests for apply_transforms."""

    def test_applies_multiple_transforms(self):
        """Test that multiple transforms are applied sequentially."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        transforms = [
            TransformConfig(op="filter", filter="x > 2"),
            TransformConfig(op="calculate", expr="x * 10", as_field="x10"),
        ]
        result = apply_transforms(df, transforms)
        assert len(result) == 3
        assert list(result["x10"]) == [30, 40, 50]

    def test_empty_transforms_returns_original(self):
        """Test that empty transforms list returns original dataframe."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_transforms(df, [])
        assert list(result["x"]) == [1, 2, 3]


class TestParseFunctionExpr:
    """Tests for _parse_function_expr."""

    def test_parses_valid_expression(self):
        """Test parsing a valid function expression."""
        func, field = _parse_function_expr("sum(value)", expected_arg=True)
        assert func == "sum"
        assert field == "value"

    def test_invalid_expression_raises(self):
        """Test that invalid expression raises error."""
        with pytest.raises(ConfigValidationError, match="must be of the form"):
            _parse_function_expr("invalid", expected_arg=True)

    def test_missing_arg_when_expected_raises(self):
        """Test that missing argument raises error when expected."""
        with pytest.raises(ConfigValidationError, match="must supply an argument"):
            _parse_function_expr("func()", expected_arg=True)


class TestDeriveFoldNames:
    """Tests for _derive_fold_names."""

    def test_default_names(self):
        """Test default names when as_fields is None."""
        var_name, value_name = _derive_fold_names(None)
        assert var_name == "key"
        assert value_name == "value"

    def test_single_as_field(self):
        """Test with single as_field."""
        var_name, value_name = _derive_fold_names(["custom_key"])
        assert var_name == "custom_key"
        assert value_name == "value"

    def test_two_as_fields(self):
        """Test with two as_fields."""
        var_name, value_name = _derive_fold_names(["custom_key", "custom_value"])
        assert var_name == "custom_key"
        assert value_name == "custom_value"
