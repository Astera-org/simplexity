"""Tests for visualization submodules to improve coverage."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from simplexity.activations.visualization.data_structures import PreparedMetadata
from simplexity.activations.visualization.dataframe_builders import (
    _build_dataframe,
    _build_dataframe_for_mappings,
    _build_metadata_columns,
    _build_scalar_dataframe,
    _build_scalar_series_dataframe,
    _extract_base_column_name,
    _infer_scalar_series_indices,
    _scalar_series_metadata,
)
from simplexity.activations.visualization.field_resolution import (
    _lookup_projection_array,
    _lookup_scalar_value,
    _maybe_component,
    _resolve_belief_states,
    _resolve_field,
)
from simplexity.activations.visualization.pattern_expansion import (
    _expand_belief_factor_mapping,
    _expand_field_mapping,
    _expand_pattern_to_indices,
    _expand_projection_key_pattern,
    _expand_scalar_pattern_ranges,
    _get_component_count,
    _parse_component_spec,
)
from simplexity.activations.visualization.preprocessing import (
    _apply_preprocessing,
    _combine_rgb,
    _expand_preprocessing_fields,
    _project_to_simplex,
)
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationDataMapping,
    ActivationVisualizationFieldRef,
    ActivationVisualizationPreprocessStep,
    CombinedMappingSection,
    ScalarSeriesMapping,
)
from simplexity.exceptions import ConfigValidationError


# pylint: disable=too-many-public-methods
class TestFieldResolution:
    """Tests for field_resolution.py functions."""

    def test_lookup_projection_array_none_key(self):
        """Test that None key raises error."""
        with pytest.raises(ConfigValidationError, match="must supply a `key` value"):
            _lookup_projection_array({}, "layer_0", None, False)

    def test_lookup_projection_array_not_found(self):
        """Test that missing projection raises error."""
        projections = {"layer_0_other": np.array([1, 2, 3])}
        with pytest.raises(ConfigValidationError, match="not available for layer"):
            _lookup_projection_array(projections, "layer_0", "missing", False)

    def test_lookup_projection_array_concat_layers_exact_match(self):
        """Test exact key match with concat_layers."""
        projections = {"my_key": np.array([1, 2, 3])}
        result = _lookup_projection_array(projections, "layer_0", "my_key", True)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_lookup_projection_array_concat_layers_suffix_match(self):
        """Test suffix match with concat_layers."""
        projections = {"prefix_my_key": np.array([4, 5, 6])}
        result = _lookup_projection_array(projections, "layer_0", "my_key", True)
        np.testing.assert_array_equal(result, [4, 5, 6])

    def test_lookup_scalar_value_concat_layers_exact(self):
        """Test scalar lookup with concat_layers exact match."""
        scalars = {"my_scalar": 0.5}
        result = _lookup_scalar_value(scalars, "layer_0", "my_scalar", True)
        assert result == 0.5

    def test_lookup_scalar_value_concat_layers_suffix(self):
        """Test scalar lookup with concat_layers suffix match."""
        scalars = {"prefix_my_scalar": 0.7}
        result = _lookup_scalar_value(scalars, "layer_0", "my_scalar", True)
        assert result == 0.7

    def test_lookup_scalar_value_not_found(self):
        """Test that missing scalar raises error."""
        with pytest.raises(ConfigValidationError, match="not available for layer"):
            _lookup_scalar_value({"other": 1.0}, "layer_0", "missing", False)

    def test_maybe_component_1d_with_component(self):
        """Test that 1D array with component raises error."""
        with pytest.raises(ConfigValidationError, match="invalid for 1D"):
            _maybe_component(np.array([1, 2, 3]), 0)

    def test_maybe_component_wrong_dim(self):
        """Test that 3D array raises error."""
        with pytest.raises(ConfigValidationError, match="must be 1D or 2D"):
            _maybe_component(np.ones((2, 3, 4)), None)

    def test_maybe_component_2d_no_component(self):
        """Test that 2D array without component raises error."""
        with pytest.raises(ConfigValidationError, match="must specify `component`"):
            _maybe_component(np.ones((3, 4)), None)

    def test_maybe_component_out_of_bounds(self):
        """Test that out of bounds component raises error."""
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            _maybe_component(np.ones((3, 4)), 10)

    def test_resolve_belief_states_wrong_dim(self):
        """Test that 1D belief states raise error."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="must be 2D or 3D"):
            _resolve_belief_states(np.array([1, 2, 3]), ref)

    def test_resolve_belief_states_3d_no_factor(self):
        """Test that 3D beliefs without factor raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=None)
        with pytest.raises(ConfigValidationError, match="no `factor` was specified"):
            _resolve_belief_states(np.ones((5, 3, 4)), ref)

    def test_resolve_belief_states_2d_with_factor(self):
        """Test that 2D beliefs with factor raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0)
        with pytest.raises(ConfigValidationError, match="Factor selection requires 3D"):
            _resolve_belief_states(np.ones((5, 4)), ref)

    def test_resolve_belief_states_factor_out_of_bounds(self):
        """Test that out of bounds factor raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=10)
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            _resolve_belief_states(np.ones((5, 3, 4)), ref)

    def test_resolve_belief_states_component_out_of_bounds(self):
        """Test that out of bounds component raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component=10)
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            _resolve_belief_states(np.ones((5, 4)), ref)

    def test_resolve_field_metadata_existing_key(self):
        """Test metadata source with existing key."""
        ref = ActivationVisualizationFieldRef(source="metadata", key="sample_index")
        metadata = {"sample_index": np.array([0, 1, 2])}
        result = _resolve_field(ref, "layer_0", {}, {}, None, False, 3, metadata)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_resolve_field_metadata_layer(self):
        """Test metadata source with layer key."""
        ref = ActivationVisualizationFieldRef(source="metadata", key="layer")
        result = _resolve_field(ref, "layer_0", {}, {}, None, False, 3, {})
        assert list(result) == ["layer_0", "layer_0", "layer_0"]

    def test_resolve_field_metadata_missing_key(self):
        """Test metadata source with missing key."""
        ref = ActivationVisualizationFieldRef(source="metadata", key="missing")
        with pytest.raises(ConfigValidationError, match="not available"):
            _resolve_field(ref, "layer_0", {}, {}, None, False, 3, {})

    def test_resolve_field_weights_missing(self):
        """Test weights source when not available."""
        ref = ActivationVisualizationFieldRef(source="weights")
        with pytest.raises(ConfigValidationError, match="unavailable"):
            _resolve_field(ref, "layer_0", {}, {}, None, False, 3, {})

    def test_resolve_field_belief_states_missing(self):
        """Test belief_states source when not available."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="were not retained"):
            _resolve_field(ref, "layer_0", {}, {}, None, False, 3, {})

    def test_resolve_field_scalars_success(self):
        """Test scalars source returns repeated value."""
        ref = ActivationVisualizationFieldRef(source="scalars", key="my_scalar")
        scalars = {"layer_0_my_scalar": 0.42}
        result = _resolve_field(ref, "layer_0", {}, scalars, None, False, 3, {})
        np.testing.assert_array_equal(result, [0.42, 0.42, 0.42])

    def test_resolve_field_unsupported_source(self):
        """Test unsupported source raises error."""
        ref = ActivationVisualizationFieldRef(source=cast(Any, "unknown"))
        with pytest.raises(ConfigValidationError, match="Unsupported field source"):
            _resolve_field(ref, "layer_0", {}, {}, None, False, 3, {})


# pylint: disable=too-many-public-methods
class TestPatternExpansion:
    """Tests for pattern_expansion.py functions."""

    def test_parse_component_spec_invalid_range_parts(self):
        """Test that malformed range raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("1...2...3")

    def test_parse_component_spec_range_not_ascending(self):
        """Test that descending range raises error."""
        with pytest.raises(ConfigValidationError, match="start must be < end"):
            _parse_component_spec("5...3")

    def test_parse_component_spec_non_numeric_range(self):
        """Test that non-numeric range raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("a...b")

    def test_parse_component_spec_unrecognized(self):
        """Test that unrecognized pattern raises error."""
        with pytest.raises(ConfigValidationError, match="Unrecognized component pattern"):
            _parse_component_spec("invalid")

    def test_expand_pattern_to_indices_no_pattern(self):
        """Test that pattern without wildcards raises error."""
        with pytest.raises(ConfigValidationError, match="has no wildcard or range"):
            _expand_pattern_to_indices("plain_key", ["key_0", "key_1"])

    def test_expand_pattern_to_indices_no_matches(self):
        """Test that no matches raises error."""
        with pytest.raises(ConfigValidationError, match="No keys found"):
            _expand_pattern_to_indices("missing_*", ["key_0", "key_1"])

    def test_expand_pattern_to_indices_non_numeric_ignored(self):
        """Test that non-numeric matches are ignored."""
        keys = ["item_0", "item_1", "item_abc"]
        result = _expand_pattern_to_indices("item_*", keys)
        assert result == [0, 1]

    def test_get_component_count_projection_success(self):
        """Test getting component count from 2D projection."""
        ref = ActivationVisualizationFieldRef(source="projections", key="proj", component="*")
        projections = {"layer_0_proj": np.ones((10, 5))}
        result = _get_component_count(ref, "layer_0", projections, None, False)
        assert result == 5

    def test_get_component_count_1d_projection(self):
        """Test that 1D projection raises error for expansion."""
        ref = ActivationVisualizationFieldRef(source="projections", key="proj")
        projections = {"layer_0_proj": np.array([1, 2, 3])}
        with pytest.raises(ConfigValidationError, match="Cannot expand 1D"):
            _get_component_count(ref, "layer_0", projections, None, False)

    def test_get_component_count_belief_states_missing(self):
        """Test that missing belief states raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="not available"):
            _get_component_count(ref, "layer_0", {}, None, False)

    def test_get_component_count_belief_states_wrong_dim(self):
        """Test that non-2D belief states raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="must be 2D"):
            _get_component_count(ref, "layer_0", {}, np.ones((2, 3, 4)), False)

    def test_get_component_count_unsupported_source(self):
        """Test that unsupported source raises error."""
        ref = ActivationVisualizationFieldRef(source="metadata", key="test")
        with pytest.raises(ConfigValidationError, match="not supported"):
            _get_component_count(ref, "layer_0", {}, None, False)

    def test_expand_projection_key_pattern_invalid(self):
        """Test that invalid key pattern raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid key pattern"):
            _expand_projection_key_pattern("plain_key", "layer_0", {}, False)

    def test_expand_projection_key_pattern_invalid_range(self):
        """Test that invalid range in key pattern raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _expand_projection_key_pattern("key_5...3", "layer_0", {}, False)

    def test_expand_projection_key_pattern_no_matches(self):
        """Test that no matching projections raises error."""
        projections = {"layer_0_other": np.ones((3, 4))}
        with pytest.raises(ConfigValidationError, match="No projection keys found"):
            _expand_projection_key_pattern("key_*", "layer_0", projections, False)

    def test_expand_belief_factor_mapping_wrong_dim(self):
        """Test that non-3D beliefs for factor expansion raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0, component=0)
        # Manually set factor to pattern string to bypass validation
        object.__setattr__(ref, "factor", "*")
        with pytest.raises(ConfigValidationError, match="require 3D beliefs"):
            _expand_belief_factor_mapping("field_*", ref, np.ones((5, 4)))

    def test_expand_belief_factor_mapping_invalid_factor(self):
        """Test that invalid factor pattern raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0, component=0)
        # Manually set factor to invalid string to bypass validation
        object.__setattr__(ref, "factor", "invalid")
        with pytest.raises(ConfigValidationError, match="Invalid factor pattern"):
            _expand_belief_factor_mapping("field_*", ref, np.ones((5, 3, 4)))

    def test_expand_belief_factor_mapping_factor_out_of_bounds(self):
        """Test that out of bounds factor range raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor="0...10", group_as="factor")
        with pytest.raises(ConfigValidationError, match="exceeds available factors"):
            _expand_belief_factor_mapping("field_*", ref, np.ones((5, 3, 4)))

    def test_expand_belief_factor_mapping_component_out_of_bounds(self):
        """Test that out of bounds component range raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor="*", component="0...10", group_as="factor")
        with pytest.raises(ConfigValidationError, match="exceeds states"):
            _expand_belief_factor_mapping("f_*_c_*", ref, np.ones((5, 2, 4)))

    def test_expand_scalar_pattern_ranges_invalid(self):
        """Test that invalid range in scalar pattern raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid range pattern"):
            _expand_scalar_pattern_ranges("metric_5...3")

    def test_expand_field_mapping_projection_no_field_pattern(self):
        """Test projection key pattern without field pattern raises error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="factor_*", group_as="factor")
        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("plain_field", ref, "layer_0", {}, {}, None, False)

    def test_expand_field_mapping_projection_too_many_patterns(self):
        """Test projection with too many field patterns raises error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="factor_*", group_as="factor")
        with pytest.raises(ConfigValidationError, match="too many patterns"):
            _expand_field_mapping("f_*_g_*_h_*", ref, "layer_0", {}, {}, None, False)

    def test_expand_field_mapping_belief_no_field_pattern(self):
        """Test belief factor pattern without field pattern raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor="*", group_as="factor")
        beliefs = np.ones((5, 3, 4))
        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("plain_field", ref, "layer_0", {}, {}, beliefs, False)

    def test_expand_field_mapping_belief_too_many_patterns(self):
        """Test belief with too many field patterns raises error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor="*", group_as="factor")
        beliefs = np.ones((5, 3, 4))
        with pytest.raises(ConfigValidationError, match="too many patterns"):
            _expand_field_mapping("f_*_g_*_h_*", ref, "layer_0", {}, {}, beliefs, False)

    def test_expand_field_mapping_scalar_field_pattern_no_key_pattern(self):
        """Test scalar with field pattern but no key pattern raises error."""
        ref = ActivationVisualizationFieldRef(source="scalars", key="plain_key")
        with pytest.raises(ConfigValidationError, match="has pattern but scalar key has no pattern"):
            _expand_field_mapping("field_*", ref, "layer_0", {}, {"plain_key": 1.0}, None, False)

    def test_expand_field_mapping_scalar_key_pattern_no_field_pattern(self):
        """Test scalar with key pattern but no field pattern raises error."""
        ref = ActivationVisualizationFieldRef(source="scalars", key="metric_*")
        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("plain_field", ref, "layer_0", {}, {"metric_0": 1.0}, None, False)


class TestPreprocessing:
    """Tests for preprocessing.py functions."""

    def test_expand_preprocessing_fields_no_matches(self):
        """Test that wildcard with no matches raises error."""
        with pytest.raises(ConfigValidationError, match="did not match any columns"):
            _expand_preprocessing_fields(["missing_*"], ["col_a", "col_b"])

    def test_expand_preprocessing_fields_range_missing_column(self):
        """Test that range expanding to missing column raises error."""
        with pytest.raises(ConfigValidationError, match="column not found"):
            _expand_preprocessing_fields(["col_0...3"], ["col_0", "col_1"])

    def test_apply_preprocessing_output_pattern_error(self):
        """Test that output field with pattern raises error."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        step = ActivationVisualizationPreprocessStep(
            type="project_to_simplex", input_fields=["a", "b", "c"], output_fields=["out_*", "out_y"]
        )
        with pytest.raises(ConfigValidationError, match="cannot contain patterns"):
            _apply_preprocessing(df, [step])

    def test_apply_preprocessing_output_range_pattern_error(self):
        """Test that output field with range pattern raises error."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        step = ActivationVisualizationPreprocessStep(
            type="project_to_simplex", input_fields=["a", "b", "c"], output_fields=["out_0...3", "out_y"]
        )
        with pytest.raises(ConfigValidationError, match="cannot contain patterns"):
            _apply_preprocessing(df, [step])

    def test_project_to_simplex_missing_column(self):
        """Test that missing column raises error."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        step = ActivationVisualizationPreprocessStep(
            type="project_to_simplex", input_fields=["a", "b", "missing"], output_fields=["x", "y"]
        )
        with pytest.raises(ConfigValidationError, match="missing from the dataframe"):
            _project_to_simplex(df, step)

    def test_project_to_simplex_success(self):
        """Test successful simplex projection."""
        df = pd.DataFrame({"p0": [0.5, 0.3], "p1": [0.3, 0.4], "p2": [0.2, 0.3]})
        step = ActivationVisualizationPreprocessStep(
            type="project_to_simplex", input_fields=["p0", "p1", "p2"], output_fields=["x", "y"]
        )
        result = _project_to_simplex(df, step)
        assert "x" in result.columns
        assert "y" in result.columns
        # x = p1 + 0.5 * p2
        np.testing.assert_allclose(result["x"], [0.3 + 0.1, 0.4 + 0.15])
        # y = sqrt(3)/2 * p2
        np.testing.assert_allclose(result["y"], [0.2 * np.sqrt(3) / 2, 0.3 * np.sqrt(3) / 2])

    def test_combine_rgb_wrong_output_count(self):
        """Test that combine_rgb with wrong output count raises error."""
        df = pd.DataFrame({"r": [0.5], "g": [0.5], "b": [0.5]})
        # Create step manually to bypass validation
        step = ActivationVisualizationPreprocessStep.__new__(ActivationVisualizationPreprocessStep)
        object.__setattr__(step, "type", "combine_rgb")
        object.__setattr__(step, "input_fields", ["r", "g", "b"])
        object.__setattr__(step, "output_fields", ["color1", "color2"])
        with pytest.raises(ConfigValidationError, match="exactly one output_field"):
            _combine_rgb(df, step)

    def test_combine_rgb_too_few_inputs(self):
        """Test that combine_rgb with <3 inputs raises error."""
        df = pd.DataFrame({"r": [0.5], "g": [0.5]})
        # Create step manually to bypass validation
        step = ActivationVisualizationPreprocessStep.__new__(ActivationVisualizationPreprocessStep)
        object.__setattr__(step, "type", "combine_rgb")
        object.__setattr__(step, "input_fields", ["r", "g"])
        object.__setattr__(step, "output_fields", ["color"])
        with pytest.raises(ConfigValidationError, match="at least three"):
            _combine_rgb(df, step)

    def test_combine_rgb_missing_column(self):
        """Test that missing column raises error."""
        df = pd.DataFrame({"r": [0.5], "g": [0.5]})
        step = ActivationVisualizationPreprocessStep(
            type="combine_rgb", input_fields=["r", "g", "missing"], output_fields=["color"]
        )
        with pytest.raises(ConfigValidationError, match="missing from the dataframe"):
            _combine_rgb(df, step)

    def test_combine_rgb_3_inputs(self):
        """Test combine_rgb with exactly 3 inputs."""
        df = pd.DataFrame({"r": [0.0, 1.0, 0.5], "g": [0.0, 0.0, 0.5], "b": [0.0, 0.0, 0.5]})
        step = ActivationVisualizationPreprocessStep(
            type="combine_rgb", input_fields=["r", "g", "b"], output_fields=["color"]
        )
        result = _combine_rgb(df, step)
        assert result["color"].iloc[0] == "#000000"  # black
        assert result["color"].iloc[1] == "#ff0000"  # red
        assert result["color"].iloc[2] == "#808080"  # gray

    def test_combine_rgb_more_than_3_inputs_pca(self):
        """Test combine_rgb with >3 inputs triggers PCA path."""
        # Create data with 4 features
        np.random.seed(42)
        df = pd.DataFrame(
            {"f0": np.random.rand(10), "f1": np.random.rand(10), "f2": np.random.rand(10), "f3": np.random.rand(10)}
        )
        step = ActivationVisualizationPreprocessStep(
            type="combine_rgb", input_fields=["f0", "f1", "f2", "f3"], output_fields=["color"]
        )
        result = _combine_rgb(df, step)
        assert "color" in result.columns
        # All colors should be valid hex colors
        for color in result["color"]:
            assert color.startswith("#")
            assert len(color) == 7

    def test_combine_rgb_pca_few_samples(self):
        """Test combine_rgb PCA path with fewer samples than components."""
        # Create 2 samples with 4 features - PCA will have <3 components
        df = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.2, 0.8], "f2": [0.3, 0.7], "f3": [0.4, 0.6]})
        step = ActivationVisualizationPreprocessStep(
            type="combine_rgb", input_fields=["f0", "f1", "f2", "f3"], output_fields=["color"]
        )
        result = _combine_rgb(df, step)
        assert "color" in result.columns
        assert len(result) == 2

    def test_apply_preprocessing_project_to_simplex(self):
        """Test full preprocessing pipeline with project_to_simplex."""
        df = pd.DataFrame({"p0": [0.5, 0.3], "p1": [0.3, 0.4], "p2": [0.2, 0.3]})
        steps = [
            ActivationVisualizationPreprocessStep(
                type="project_to_simplex", input_fields=["p0", "p1", "p2"], output_fields=["x", "y"]
            )
        ]
        result = _apply_preprocessing(df, steps)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_apply_preprocessing_combine_rgb(self):
        """Test full preprocessing pipeline with combine_rgb."""
        df = pd.DataFrame({"r": [0.5], "g": [0.5], "b": [0.5]})
        steps = [
            ActivationVisualizationPreprocessStep(
                type="combine_rgb", input_fields=["r", "g", "b"], output_fields=["color"]
            )
        ]
        result = _apply_preprocessing(df, steps)
        assert "color" in result.columns

    def test_apply_preprocessing_with_pattern_expansion(self):
        """Test preprocessing with pattern expansion in input fields."""
        df = pd.DataFrame({"val_0": [0.2], "val_1": [0.3], "val_2": [0.5]})
        steps = [
            ActivationVisualizationPreprocessStep(
                type="project_to_simplex", input_fields=["val_*"], output_fields=["x", "y"]
            )
        ]
        result = _apply_preprocessing(df, steps)
        assert "x" in result.columns
        assert "y" in result.columns


# pylint: disable=too-many-public-methods
class TestDataframeBuilders:
    """Tests for dataframe_builders.py functions."""

    def test_extract_base_column_name_with_group_pattern(self):
        """Test extracting base column name with group value pattern."""
        result = _extract_base_column_name("factor_0_projected", "0", "factor_*")
        assert result == "projected"

    def test_extract_base_column_name_no_pattern(self):
        """Test extracting base column name when no pattern."""
        result = _extract_base_column_name("my_column", "0", None)
        assert result == "my_column"

    def test_extract_base_column_name_no_match(self):
        """Test extracting base column name when pattern doesn't match."""
        result = _extract_base_column_name("other_column", "0", "factor_*")
        assert result == "other_column"

    def test_scalar_series_metadata_with_arrays(self):
        """Test extracting metadata from arrays."""
        metadata = {"step": np.array([10]), "name": np.array(["test"])}
        result = _scalar_series_metadata(metadata)
        assert result["step"] == 10
        assert result["name"] == "test"

    def test_scalar_series_metadata_with_empty_array(self):
        """Test that empty arrays are skipped."""
        metadata = {"step": np.array([10]), "empty": np.array([])}
        result = _scalar_series_metadata(metadata)
        assert result["step"] == 10
        assert "empty" not in result

    def test_scalar_series_metadata_with_scalar(self):
        """Test extracting metadata from scalar values."""
        metadata = {"step": 10, "name": "test"}
        result = _scalar_series_metadata(metadata)
        assert result["step"] == 10
        assert result["name"] == "test"

    def test_infer_scalar_series_indices_success(self):
        """Test inferring scalar series indices from available keys."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        scalars = {
            "analysis/layer_0_cumvar_0": 0.5,
            "analysis/layer_0_cumvar_1": 0.7,
            "analysis/layer_0_cumvar_2": 0.9,
        }
        result = _infer_scalar_series_indices(mapping, scalars, "layer_0", "analysis")
        assert result == [0, 1, 2]

    def test_infer_scalar_series_indices_empty_body(self):
        """Test that empty body between prefix and suffix is skipped."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_pc{index}_var", index_field="component", value_field="variance"
        )
        # Key that matches prefix and suffix but has empty body
        scalars = {
            "analysis/layer_0_pc_var": 0.5,  # Empty between pc and _var
            "analysis/layer_0_pc0_var": 0.3,
        }
        result = _infer_scalar_series_indices(mapping, scalars, "layer_0", "analysis")
        assert result == [0]  # Only numeric index included

    def test_infer_scalar_series_indices_no_matches(self):
        """Test that no matching indices raises error."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        scalars = {"analysis/other_metric": 1.0}
        with pytest.raises(ConfigValidationError, match="could not infer indices"):
            _infer_scalar_series_indices(mapping, scalars, "layer_0", "analysis")

    def test_infer_scalar_series_indices_with_suffix(self):
        """Test inferring indices when template has suffix after index."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_pc{index}_var", index_field="component", value_field="variance"
        )
        scalars = {
            "analysis/layer_0_pc0_var": 0.5,
            "analysis/layer_0_pc1_var": 0.3,
            "analysis/layer_0_pc2_var": 0.2,
            "analysis/layer_0_other": 1.0,  # Should not match
        }
        result = _infer_scalar_series_indices(mapping, scalars, "layer_0", "analysis")
        assert result == [0, 1, 2]

    def test_infer_scalar_series_indices_non_numeric_skipped(self):
        """Test that non-numeric values are skipped."""
        mapping = ScalarSeriesMapping(key_template="{layer}_item_{index}", index_field="idx", value_field="val")
        scalars = {
            "analysis/layer_0_item_0": 0.5,
            "analysis/layer_0_item_abc": 0.7,  # Non-numeric, should be skipped
            "analysis/layer_0_item_1": 0.9,
        }
        result = _infer_scalar_series_indices(mapping, scalars, "layer_0", "analysis")
        assert result == [0, 1]

    def test_build_scalar_series_dataframe_success(self):
        """Test building scalar series dataframe."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        metadata = {"step": np.array([10]), "analysis": np.array(["pca"])}
        scalars = {
            "analysis/layer_0_cumvar_0": 0.5,
            "analysis/layer_0_cumvar_1": 0.7,
            "analysis/layer_1_cumvar_0": 0.6,
        }
        result = _build_scalar_series_dataframe(mapping, metadata, scalars, ["layer_0", "layer_1"], "analysis")
        assert len(result) == 3
        assert "component" in result.columns
        assert "cumvar" in result.columns
        assert "layer" in result.columns

    def test_build_scalar_series_dataframe_no_matches(self):
        """Test that no matching scalars raises error."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        metadata = {"step": np.array([10])}
        scalars = {"analysis/other_metric": 1.0}
        # Error comes from _infer_scalar_series_indices when no indices are found
        with pytest.raises(ConfigValidationError, match="could not infer indices"):
            _build_scalar_series_dataframe(mapping, metadata, scalars, ["layer_0"], "analysis")

    def test_build_scalar_series_dataframe_with_explicit_indices(self):
        """Test building scalar series dataframe with explicit index_values."""
        mapping = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar", index_values=[0, 1]
        )
        metadata = {"step": np.array([10])}
        scalars = {
            "analysis/layer_0_cumvar_0": 0.5,
            "analysis/layer_0_cumvar_1": 0.7,
            "analysis/layer_0_cumvar_2": 0.9,  # Not in index_values, should be skipped
        }
        result = _build_scalar_series_dataframe(mapping, metadata, scalars, ["layer_0"], "analysis")
        assert len(result) == 2
        assert list(result["component"]) == [0, 1]

    def test_build_scalar_dataframe_scalar_pattern(self):
        """Test building scalar dataframe with scalar_pattern source."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="layer_*_rmse")}
        scalars = {
            "analysis/layer_0_rmse": 0.1,
            "analysis/layer_1_rmse": 0.2,
        }
        result = _build_scalar_dataframe(mappings, scalars, {}, "analysis", 5)
        assert len(result) == 2
        assert "step" in result.columns
        assert "rmse" in result.columns
        assert all(result["step"] == 5)

    def test_build_scalar_dataframe_scalar_history(self):
        """Test building scalar dataframe with scalar_history source."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_history", key="metric")}
        scalars = {}
        scalar_history = {"analysis/metric": [(0, 0.5), (10, 0.3), (20, 0.1)]}
        result = _build_scalar_dataframe(mappings, scalars, scalar_history, "analysis", 20)
        assert len(result) == 3
        assert list(result["step"]) == [0, 10, 20]

    def test_build_scalar_dataframe_scalar_history_fallback(self):
        """Test scalar_history falls back to current scalars when no history."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_history", key="metric")}
        scalars = {"analysis/metric": 0.42}
        scalar_history = {}
        result = _build_scalar_dataframe(mappings, scalars, scalar_history, "analysis", 5)
        assert len(result) == 1
        assert result["step"].iloc[0] == 5
        assert result["rmse"].iloc[0] == 0.42

    def test_build_scalar_dataframe_no_matches(self):
        """Test that no matching scalars raises error."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="missing_*")}
        scalars = {"analysis/other": 1.0}
        with pytest.raises(ConfigValidationError, match="No scalar pattern keys found"):
            _build_scalar_dataframe(mappings, scalars, {}, "analysis", 5)

    def test_build_scalar_dataframe_non_scalar_source_skipped(self):
        """Test that non-scalar sources are skipped."""
        mappings = {
            "proj": ActivationVisualizationFieldRef(source="projections", key="my_proj"),
            "rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="layer_*_rmse"),
        }
        scalars = {"analysis/layer_0_rmse": 0.1}
        result = _build_scalar_dataframe(mappings, scalars, {}, "analysis", 5)
        # Only scalar_pattern should be in result
        assert "rmse" in result.columns
        assert len(result) == 1

    def test_build_scalar_dataframe_simple_key(self):
        """Test scalar_pattern with non-pattern key."""
        # Use field name "value" to avoid conflict with hardcoded "metric" column
        mappings = {"value": ActivationVisualizationFieldRef(source="scalar_pattern", key="my_metric")}
        scalars = {"analysis/my_metric": 0.42}
        result = _build_scalar_dataframe(mappings, scalars, {}, "analysis", 10)
        assert len(result) == 1
        assert result["value"].iloc[0] == 0.42
        assert result["metric"].iloc[0] == "analysis/my_metric"  # Check the metric key column

    def test_build_scalar_dataframe_key_none(self):
        """Test that scalar_pattern with key=None raises error."""
        ref = ActivationVisualizationFieldRef(source="scalar_pattern", key="placeholder")
        # Bypass validation to set key to None
        object.__setattr__(ref, "key", None)
        mappings = {"value": ref}
        with pytest.raises(ConfigValidationError, match="must specify a key"):
            _build_scalar_dataframe(mappings, {"analysis/test": 1.0}, {}, "analysis", 5)

    def test_build_scalar_dataframe_no_matching_values(self):
        """Test that no matching values raises error with pattern."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="layer_*_missing")}
        # Scalars exist but don't match the pattern
        scalars = {"analysis/layer_0_other": 0.1, "analysis/something_else": 0.2}
        with pytest.raises(ConfigValidationError, match="No scalar pattern keys found"):
            _build_scalar_dataframe(mappings, scalars, {}, "analysis", 5)

    def test_build_metadata_columns(self):
        """Test building metadata columns."""

        sequences: list[tuple[int, ...]] = [(1, 2, 3), (4, 5)]
        steps = np.array([3, 2])
        metadata = PreparedMetadata(sequences=sequences, steps=steps, select_last_token=False)
        weights = np.array([1.0, 0.5])
        result = _build_metadata_columns("my_analysis", metadata, weights)
        assert "analysis" in result
        assert "step" in result
        assert "sequence_length" in result
        assert "sequence" in result
        assert "sample_index" in result
        assert "weight" in result
        assert list(result["analysis"]) == ["my_analysis", "my_analysis"]
        assert list(result["step"]) == [3, 2]
        assert list(result["weight"]) == [1.0, 0.5]

    def test_build_dataframe_for_mappings_simple(self):
        """Test _build_dataframe_for_mappings with simple projection mapping."""
        mappings = {"x": ActivationVisualizationFieldRef(source="projections", key="pca", component=0)}
        metadata = {"step": np.array([1, 2]), "analysis": np.array(["test", "test"])}
        projections = {"layer_0_pca": np.array([[0.1, 0.2], [0.3, 0.4]])}
        result = _build_dataframe_for_mappings(mappings, metadata, projections, {}, None, False, ["layer_0"])
        assert "x" in result.columns
        assert "layer" in result.columns
        assert len(result) == 2

    def test_build_dataframe_for_mappings_belief_only(self):
        """Test _build_dataframe_for_mappings with belief_states only (no layer iteration)."""
        mappings = {"belief": ActivationVisualizationFieldRef(source="belief_states", component=0)}
        metadata = {"step": np.array([1, 2])}
        beliefs = np.array([[0.8, 0.2], [0.6, 0.4]])
        result = _build_dataframe_for_mappings(mappings, metadata, {}, {}, beliefs, False, ["layer_0"])
        assert "belief" in result.columns
        assert len(result) == 2
        # Belief-only mode uses "_no_layer_" placeholder
        assert result["layer"].iloc[0] == "_no_layer_"

    def test_build_dataframe_for_mappings_with_groups(self):
        """Test _build_dataframe_for_mappings with group expansion."""
        # Use belief_states with factor pattern to trigger group expansion
        # field_name has one *, factor has one *, so component expansion happens
        mappings = {
            "prob_*": ActivationVisualizationFieldRef(
                source="belief_states", factor="*", component=0, group_as="factor"
            )
        }
        metadata = {"step": np.array([1])}
        # 3D beliefs: (samples, factors, states)
        beliefs = np.array([[[0.8, 0.2], [0.6, 0.4]]])  # 1 sample, 2 factors, 2 states
        result = _build_dataframe_for_mappings(mappings, metadata, {}, {}, beliefs, False, ["layer_0"])
        assert "factor" in result.columns
        # Factor expansion creates separate prob_0 and prob_1 columns
        assert "prob_0" in result.columns or "prob_1" in result.columns
        # Should have 2 rows (one per factor group)
        assert len(result) == 2

    def test_build_dataframe_for_mappings_error_wrapping(self):
        """Test that errors from _expand_field_mapping are wrapped with context."""
        # Create a mapping with a key pattern that will fail expansion due to no matching projections
        # The key "factor_*" is a pattern that needs expansion, which fails when no projections match
        mappings = {"x_*": ActivationVisualizationFieldRef(source="projections", key="factor_*", group_as="factor")}
        metadata = {"step": np.array([1])}
        with pytest.raises(ConfigValidationError, match="Error expanding 'x_\\*' for layer"):
            _build_dataframe_for_mappings(mappings, metadata, {}, {}, None, False, ["layer_0"])

    def test_build_dataframe_with_scalar_pattern(self):
        """Test _build_dataframe with scalar_pattern source."""
        data_mapping = ActivationVisualizationDataMapping(
            mappings={"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="layer_*_rmse")}
        )
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1]), "analysis": np.array(["test"])}
        scalars = {"test/layer_0_rmse": 0.1, "test/layer_1_rmse": 0.2}
        result = _build_dataframe(viz_cfg, metadata, {}, scalars, {}, 10, None, False, ["layer_0", "layer_1"])
        assert "rmse" in result.columns
        assert len(result) == 2

    def test_build_dataframe_with_scalar_series(self):
        """Test _build_dataframe with scalar_series source."""
        scalar_series = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        data_mapping = ActivationVisualizationDataMapping(mappings={}, scalar_series=scalar_series)
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1]), "analysis": np.array(["test"])}
        scalars = {"test/layer_0_cumvar_0": 0.5, "test/layer_0_cumvar_1": 0.7}
        result = _build_dataframe(viz_cfg, metadata, {}, scalars, {}, None, None, False, ["layer_0"])
        assert "component" in result.columns
        assert "cumvar" in result.columns

    def test_build_dataframe_combined_mappings(self):
        """Test _build_dataframe with combined mappings."""
        combined = [
            CombinedMappingSection(
                label="projected",
                mappings={"x": ActivationVisualizationFieldRef(source="projections", key="pca", component=0)},
            ),
            CombinedMappingSection(
                label="raw",
                mappings={"x": ActivationVisualizationFieldRef(source="projections", key="raw", component=0)},
            ),
        ]
        data_mapping = ActivationVisualizationDataMapping(mappings={}, combined=combined, combine_as="source")
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1])}
        projections = {
            "layer_0_pca": np.array([[0.1, 0.2]]),
            "layer_0_raw": np.array([[0.5, 0.6]]),
        }
        result = _build_dataframe(viz_cfg, metadata, projections, {}, {}, None, None, False, ["layer_0"])
        assert "source" in result.columns
        assert set(result["source"]) == {"projected", "raw"}
        assert len(result) == 2

    def test_build_dataframe_scalar_pattern_no_step(self):
        """Test that scalar_pattern without step raises error."""
        data_mapping = ActivationVisualizationDataMapping(
            mappings={"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="metric")}
        )
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1]), "analysis": np.array(["test"])}
        with pytest.raises(ConfigValidationError, match="without the `step` parameter"):
            _build_dataframe(viz_cfg, metadata, {}, {"test/metric": 0.1}, {}, None, None, False, [])

    def test_build_dataframe_scalar_pattern_no_analysis(self):
        """Test that scalar_pattern without analysis metadata raises error."""
        data_mapping = ActivationVisualizationDataMapping(
            mappings={"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="metric")}
        )
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1])}  # No "analysis" key
        with pytest.raises(ConfigValidationError, match="requires 'analysis'"):
            _build_dataframe(viz_cfg, metadata, {}, {"test/metric": 0.1}, {}, 10, None, False, [])

    def test_build_dataframe_scalar_series_no_analysis(self):
        """Test that scalar_series without analysis metadata raises error."""
        scalar_series = ScalarSeriesMapping(
            key_template="{layer}_cumvar_{index}", index_field="component", value_field="cumvar"
        )
        data_mapping = ActivationVisualizationDataMapping(mappings={}, scalar_series=scalar_series)
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        metadata = {"step": np.array([1])}  # No "analysis" key
        with pytest.raises(ConfigValidationError, match="requires 'analysis'"):
            _build_dataframe(viz_cfg, metadata, {}, {}, {}, None, None, False, ["layer_0"])
