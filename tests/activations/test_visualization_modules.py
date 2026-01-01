"""Tests for visualization submodules to improve coverage."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from simplexity.activations.visualization.data_structures import PreparedMetadata
from simplexity.activations.visualization.dataframe_builders import (
    _apply_sampling,
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
    _pca_project,
    _project_to_simplex,
)
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationDataMapping,
    ActivationVisualizationFieldRef,
    ActivationVisualizationPreprocessStep,
    CombinedMappingSection,
    SamplingConfig,
    ScalarSeriesMapping,
)
from simplexity.exceptions import ConfigValidationError


# pylint: disable=too-many-public-methods
class TestFieldResolution:
    """Tests for field_resolution.py functions."""

    @pytest.mark.parametrize(
        ("projections", "key", "match"),
        [
            ({}, None, "must supply a `key` value"),
            ({"layer_0_other": np.array([1, 2, 3])}, "missing", "not available for layer"),
        ],
    )
    def test_lookup_projection_array_errors(self, projections, key, match):
        """Test that lookup_projection_array raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _lookup_projection_array(projections, "layer_0", key, False)

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

    @pytest.mark.parametrize(
        ("scalars", "key", "concat_layers", "expected"),
        [
            ({"my_scalar": 0.5}, "my_scalar", True, 0.5),
            ({"prefix_my_scalar": 0.7}, "my_scalar", True, 0.7),
        ],
    )
    def test_lookup_scalar_value_success(self, scalars, key, concat_layers, expected):
        """Test successful scalar value lookup."""
        result = _lookup_scalar_value(scalars, "layer_0", key, concat_layers)
        assert result == expected

    def test_lookup_scalar_value_not_found(self):
        """Test that missing scalar raises error."""
        with pytest.raises(ConfigValidationError, match="not available for layer"):
            _lookup_scalar_value({"other": 1.0}, "layer_0", "missing", False)

    @pytest.mark.parametrize(
        ("array", "component", "match"),
        [
            (np.array([1, 2, 3]), 0, "invalid for 1D"),
            (np.ones((2, 3, 4)), None, "must be 1D or 2D"),
            (np.ones((3, 4)), None, "must specify `component`"),
            (np.ones((3, 4)), 10, "out of bounds"),
        ],
    )
    def test_maybe_component_errors(self, array, component, match):
        """Test that maybe_component raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _maybe_component(array, component)

    @pytest.mark.parametrize(
        ("beliefs", "ref_kwargs", "match"),
        [
            (np.array([1, 2, 3]), {}, "must be 2D or 3D"),
            (np.ones((5, 3, 4)), {"factor": None}, "no `factor` was specified"),
            (np.ones((5, 4)), {"factor": 0}, "Factor selection requires 3D"),
            (np.ones((5, 3, 4)), {"factor": 10}, "out of bounds"),
            (np.ones((5, 4)), {"component": 10}, "out of bounds"),
        ],
    )
    def test_resolve_belief_states_errors(self, beliefs, ref_kwargs, match):
        """Test that resolve_belief_states raises expected errors."""
        ref = ActivationVisualizationFieldRef(source="belief_states", **ref_kwargs)
        with pytest.raises(ConfigValidationError, match=match):
            _resolve_belief_states(beliefs, ref)

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

    @pytest.mark.parametrize(
        ("source", "key", "match"),
        [
            ("metadata", "missing", "not available"),
            ("weights", None, "unavailable"),
            ("belief_states", None, "were not retained"),
        ],
    )
    def test_resolve_field_missing_sources(self, source, key, match):
        """Test that missing sources raise expected errors."""
        ref = ActivationVisualizationFieldRef(source=source, key=key)
        with pytest.raises(ConfigValidationError, match=match):
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

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("1...2...3", "Invalid range"),
            ("5...3", "start must be < end"),
            ("a...b", "Invalid range"),
            ("invalid", "Unrecognized component pattern"),
        ],
    )
    def test_parse_component_spec_errors(self, spec, match):
        """Test that parse_component_spec raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _parse_component_spec(spec)

    @pytest.mark.parametrize(
        ("pattern", "keys", "match"),
        [
            ("plain_key", ["key_0", "key_1"], "has no wildcard or range"),
            ("missing_*", ["key_0", "key_1"], "No keys found"),
        ],
    )
    def test_expand_pattern_to_indices_errors(self, pattern, keys, match):
        """Test that expand_pattern_to_indices raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _expand_pattern_to_indices(pattern, keys)

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

    @pytest.mark.parametrize(
        ("ref_kwargs", "projections", "beliefs", "match"),
        [
            ({"source": "projections", "key": "proj"}, {"layer_0_proj": np.array([1, 2, 3])}, None, "Cannot expand 1D"),
            ({"source": "belief_states"}, {}, None, "not available"),
            ({"source": "belief_states"}, {}, np.ones((2, 3, 4)), "must be 2D"),
            ({"source": "metadata", "key": "test"}, {}, None, "not supported"),
        ],
    )
    def test_get_component_count_errors(self, ref_kwargs, projections, beliefs, match):
        """Test that get_component_count raises expected errors."""
        ref = ActivationVisualizationFieldRef(**ref_kwargs)
        with pytest.raises(ConfigValidationError, match=match):
            _get_component_count(ref, "layer_0", projections, beliefs, False)

    @pytest.mark.parametrize(
        ("key_pattern", "projections", "match"),
        [
            ("plain_key", {}, "Invalid key pattern"),
            ("key_5...3", {}, "Invalid range"),
            ("key_*", {"layer_0_other": np.ones((3, 4))}, "No projection keys found"),
        ],
    )
    def test_expand_projection_key_pattern_errors(self, key_pattern, projections, match):
        """Test that expand_projection_key_pattern raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _expand_projection_key_pattern(key_pattern, "layer_0", projections, False)

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

    @pytest.mark.parametrize(
        ("fields", "columns", "match"),
        [
            (["missing_*"], ["col_a", "col_b"], "did not match any columns"),
            (["col_0...3"], ["col_0", "col_1"], "column not found"),
        ],
    )
    def test_expand_preprocessing_fields_errors(self, fields, columns, match):
        """Test that expand_preprocessing_fields raises expected errors."""
        with pytest.raises(ConfigValidationError, match=match):
            _expand_preprocessing_fields(fields, columns)

    @pytest.mark.parametrize("output_fields", [["out_*", "out_y"], ["out_0...3", "out_y"]])
    def test_apply_preprocessing_output_pattern_error(self, output_fields):
        """Test that output fields with patterns raise error."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        step = ActivationVisualizationPreprocessStep(
            type="project_to_simplex", input_fields=["a", "b", "c"], output_fields=output_fields
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

    @pytest.mark.parametrize(
        ("input_fields", "output_fields", "match"),
        [
            (["r", "g", "b"], ["color1", "color2"], "exactly one output_field"),
            (["r", "g"], ["color"], "at least three"),
        ],
    )
    def test_combine_rgb_validation_errors(self, input_fields, output_fields, match):
        """Test that combine_rgb raises expected validation errors."""
        df = pd.DataFrame({field: [0.5] for field in input_fields})
        # Create step manually to bypass validation
        step = ActivationVisualizationPreprocessStep.__new__(ActivationVisualizationPreprocessStep)
        object.__setattr__(step, "type", "combine_rgb")
        object.__setattr__(step, "input_fields", input_fields)
        object.__setattr__(step, "output_fields", output_fields)
        with pytest.raises(ConfigValidationError, match=match):
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
        """Test combine_rgb with exactly 3 inputs.

        Note: combine_rgb performs per-column min-max normalization, so to get
        expected colors we need data where each column spans [0, 1].
        """
        df = pd.DataFrame({"r": [0.0, 1.0, 0.5], "g": [0.0, 1.0, 0.5], "b": [0.0, 1.0, 0.5]})
        step = ActivationVisualizationPreprocessStep(
            type="combine_rgb", input_fields=["r", "g", "b"], output_fields=["color"]
        )
        result = _combine_rgb(df, step)
        assert result["color"].iloc[0] == "#000000"  # black
        assert result["color"].iloc[1] == "#ffffff"  # white
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

    @pytest.mark.parametrize(
        ("df_dict", "step_type", "input_fields", "output_fields", "expected_cols"),
        [
            ({"p0": [0.5, 0.3], "p1": [0.3, 0.4], "p2": [0.2, 0.3]}, "project_to_simplex", ["p0", "p1", "p2"], ["x", "y"], ["x", "y"]),
            ({"r": [0.5], "g": [0.5], "b": [0.5]}, "combine_rgb", ["r", "g", "b"], ["color"], ["color"]),
            ({"val_0": [0.2], "val_1": [0.3], "val_2": [0.5]}, "project_to_simplex", ["val_*"], ["x", "y"], ["x", "y"]),
        ],
    )
    def test_apply_preprocessing_pipeline(self, df_dict, step_type, input_fields, output_fields, expected_cols):
        """Test full preprocessing pipeline with various step types."""
        df = pd.DataFrame(df_dict)
        steps = [ActivationVisualizationPreprocessStep(type=step_type, input_fields=input_fields, output_fields=output_fields)]
        result = _apply_preprocessing(df, steps)
        for col in expected_cols:
            assert col in result.columns

    # ---- pca_project tests ----

    def test_pca_project_basic(self):
        """Test basic PCA projection (5 dims -> 3 dims)."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "f0": np.random.rand(10),
                "f1": np.random.rand(10),
                "f2": np.random.rand(10),
                "f3": np.random.rand(10),
                "f4": np.random.rand(10),
            }
        )
        step = ActivationVisualizationPreprocessStep(
            type="pca_project", input_fields=["f0", "f1", "f2", "f3", "f4"], output_fields=["pca_x", "pca_y", "pca_z"]
        )
        result = _pca_project(df, step)
        assert "pca_x" in result.columns
        assert "pca_y" in result.columns
        assert "pca_z" in result.columns
        assert len(result) == 10

    def test_pca_project_fewer_samples_than_components(self):
        """Test PCA projection with fewer samples than requested components."""
        # 2 samples, 4 features -> max 2 components, but requesting 3
        df = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.2, 0.8], "f2": [0.3, 0.7], "f3": [0.4, 0.6]})
        step = ActivationVisualizationPreprocessStep(
            type="pca_project", input_fields=["f0", "f1", "f2", "f3"], output_fields=["pca_x", "pca_y", "pca_z"]
        )
        result = _pca_project(df, step)
        assert "pca_x" in result.columns
        assert "pca_y" in result.columns
        assert "pca_z" in result.columns
        # Third component should be padded with zeros
        np.testing.assert_array_equal(result["pca_z"], [0.0, 0.0])

    def test_pca_project_more_output_than_input(self):
        """Test PCA projection requesting more outputs than inputs (should pad)."""
        # 3 features, requesting 5 outputs -> pad with zeros
        np.random.seed(42)
        df = pd.DataFrame({"f0": np.random.rand(10), "f1": np.random.rand(10), "f2": np.random.rand(10)})
        step = ActivationVisualizationPreprocessStep(
            type="pca_project",
            input_fields=["f0", "f1", "f2"],
            output_fields=["pca_0", "pca_1", "pca_2", "pca_3", "pca_4"],
        )
        result = _pca_project(df, step)
        assert "pca_0" in result.columns
        assert "pca_4" in result.columns
        # Components 3 and 4 should be zeros (only 3 input features)
        np.testing.assert_array_equal(result["pca_3"], np.zeros(10))
        np.testing.assert_array_equal(result["pca_4"], np.zeros(10))

    def test_pca_project_missing_column(self):
        """Test that missing input column raises error."""
        df = pd.DataFrame({"f0": [0.5], "f1": [0.5]})
        step = ActivationVisualizationPreprocessStep(
            type="pca_project", input_fields=["f0", "f1", "missing"], output_fields=["pca_x"]
        )
        with pytest.raises(ConfigValidationError, match="missing from the dataframe"):
            _pca_project(df, step)

    def test_pca_project_single_output(self):
        """Test PCA projection to a single dimension."""
        np.random.seed(42)
        df = pd.DataFrame({"f0": np.random.rand(10), "f1": np.random.rand(10), "f2": np.random.rand(10)})
        step = ActivationVisualizationPreprocessStep(
            type="pca_project", input_fields=["f0", "f1", "f2"], output_fields=["pca_1d"]
        )
        result = _pca_project(df, step)
        assert "pca_1d" in result.columns
        assert len(result) == 10

    @pytest.mark.parametrize(
        ("input_fields",),
        [
            (["f0", "f1", "f2", "f3"],),
            (["prob_*"],),
        ],
    )
    def test_apply_preprocessing_pca_project(self, input_fields):
        """Test full preprocessing pipeline with pca_project."""
        np.random.seed(42)
        if "*" in input_fields[0]:
            df = pd.DataFrame({f"prob_{i}": np.random.rand(10) for i in range(5)})
        else:
            df = pd.DataFrame({f: np.random.rand(10) for f in input_fields})
        steps = [
            ActivationVisualizationPreprocessStep(
                type="pca_project", input_fields=input_fields, output_fields=["pca_x", "pca_y", "pca_z"]
            )
        ]
        result = _apply_preprocessing(df, steps)
        assert "pca_x" in result.columns
        assert "pca_y" in result.columns
        assert "pca_z" in result.columns

    def test_pca_project_then_combine_rgb(self):
        """Test chaining pca_project with combine_rgb."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "prob_0": np.random.rand(10),
                "prob_1": np.random.rand(10),
                "prob_2": np.random.rand(10),
                "prob_3": np.random.rand(10),
                "prob_4": np.random.rand(10),
            }
        )
        steps = [
            ActivationVisualizationPreprocessStep(
                type="pca_project", input_fields=["prob_*"], output_fields=["pca_x", "pca_y", "pca_z"]
            ),
            ActivationVisualizationPreprocessStep(
                type="combine_rgb", input_fields=["pca_x", "pca_y", "pca_z"], output_fields=["point_color"]
            ),
        ]
        result = _apply_preprocessing(df, steps)
        assert "pca_x" in result.columns
        assert "pca_y" in result.columns
        assert "pca_z" in result.columns
        assert "point_color" in result.columns
        # All colors should be valid hex colors
        for color in result["point_color"]:
            assert color.startswith("#")
            assert len(color) == 7


# pylint: disable=too-many-public-methods
class TestDataframeBuilders:
    """Tests for dataframe_builders.py functions."""

    @pytest.mark.parametrize(
        ("column_name", "group_value", "expected"),
        [
            ("factor_0_projected", "0", "projected"),
            ("my_column", "0", "my_column"),
            ("other_column", "0", "other_column"),
        ],
    )
    def test_extract_base_column_name(self, column_name, group_value, expected):
        """Test extracting base column name."""
        result = _extract_base_column_name(column_name, group_value)
        assert result == expected

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({"step": np.array([10]), "name": np.array(["test"])}, {"step": 10, "name": "test"}),
            ({"step": np.array([10]), "empty": np.array([])}, {"step": 10}),
            ({"step": 10, "name": "test"}, {"step": 10, "name": "test"}),
        ],
    )
    def test_scalar_series_metadata(self, metadata, expected):
        """Test extracting metadata from various inputs."""
        result = _scalar_series_metadata(metadata)
        for key, value in expected.items():
            assert result[key] == value
        for key in metadata:
            if key not in expected:
                assert key not in result

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
        scalars = {"analysis/layer_0_rmse": 0.1, "analysis/layer_1_rmse": 0.2}
        result = _build_scalar_dataframe(mappings, scalars, {}, "analysis", 5)
        assert len(result) == 2
        assert "step" in result.columns and "rmse" in result.columns and all(result["step"] == 5)

    @pytest.mark.parametrize(
        ("scalar_history", "scalars", "expected_len", "expected_steps"),
        [
            ({"analysis/metric": [(0, 0.5), (10, 0.3), (20, 0.1)]}, {}, 3, [0, 10, 20]),
            ({}, {"analysis/metric": 0.42}, 1, [5]),
        ],
    )
    def test_build_scalar_dataframe_scalar_history(self, scalar_history, scalars, expected_len, expected_steps):
        """Test building scalar dataframe with scalar_history source."""
        mappings = {"rmse": ActivationVisualizationFieldRef(source="scalar_history", key="metric")}
        step = 20 if scalar_history else 5
        result = _build_scalar_dataframe(mappings, scalars, scalar_history, "analysis", step)
        assert len(result) == expected_len
        assert list(result["step"]) == expected_steps

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
        mappings = {"value": ActivationVisualizationFieldRef(source="scalar_pattern", key="my_metric")}
        result = _build_scalar_dataframe(mappings, {"analysis/my_metric": 0.42}, {}, "analysis", 10)
        assert len(result) == 1 and result["value"].iloc[0] == 0.42 and result["metric"].iloc[0] == "analysis/my_metric"

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
        metadata = PreparedMetadata(sequences=sequences, steps=np.array([3, 2]), select_last_token=False)
        result = _build_metadata_columns("my_analysis", metadata, np.array([1.0, 0.5]))
        for col in ["analysis", "step", "sequence_length", "sequence", "sample_index", "weight"]:
            assert col in result
        assert list(result["analysis"]) == ["my_analysis", "my_analysis"]
        assert list(result["step"]) == [3, 2] and list(result["weight"]) == [1.0, 0.5]

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

    @pytest.mark.parametrize(
        ("metadata", "step", "match"),
        [
            ({"step": np.array([1]), "analysis": np.array(["test"])}, None, "without the `step` parameter"),
            ({"step": np.array([1])}, 10, "requires 'analysis'"),
        ],
    )
    def test_build_dataframe_scalar_pattern_validation(self, metadata, step, match):
        """Test that scalar_pattern validates required parameters."""
        data_mapping = ActivationVisualizationDataMapping(
            mappings={"rmse": ActivationVisualizationFieldRef(source="scalar_pattern", key="metric")}
        )
        viz_cfg = ActivationVisualizationConfig(name="test", data_mapping=data_mapping)
        with pytest.raises(ConfigValidationError, match=match):
            _build_dataframe(viz_cfg, metadata, {}, {"test/metric": 0.1}, {}, step, None, False, [])

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


class TestSampling:
    """Tests for DataFrame sampling functionality."""

    @pytest.mark.parametrize(
        ("df_size", "max_points", "expected_len"),
        [
            (100, 20, 20),  # Should sample
            (10, 20, 10),  # Should not sample
        ],
    )
    def test_sampling_basic(self, df_size, max_points, expected_len):
        """Test basic sampling behavior."""
        df = pd.DataFrame({"a": range(df_size), "b": range(df_size)})
        config = SamplingConfig(max_points=max_points, seed=42)
        result = _apply_sampling(df, config, facet_columns=[])
        assert len(result) == expected_len

    def test_sampling_per_facet_group(self):
        """Test that sampling applies per facet group."""
        df = pd.DataFrame(
            {
                "factor": ["0"] * 50 + ["1"] * 50 + ["2"] * 50,
                "value": range(150),
            }
        )
        config = SamplingConfig(max_points=10, seed=42)
        result = _apply_sampling(df, config, facet_columns=["factor"])

        assert len(result) == 30  # 10 per factor * 3 factors
        for factor in ["0", "1", "2"]:
            factor_count = len(result[result["factor"] == factor])
            assert factor_count == 10

    def test_sampling_multiple_facet_columns(self):
        """Test sampling with multiple facet columns."""
        df = pd.DataFrame(
            {
                "layer": ["layer_0"] * 40 + ["layer_1"] * 40,
                "factor": (["0"] * 20 + ["1"] * 20) * 2,
                "value": range(80),
            }
        )
        config = SamplingConfig(max_points=5, seed=42)
        result = _apply_sampling(df, config, facet_columns=["layer", "factor"])

        # Should have 4 groups (2 layers * 2 factors), each with max 5 points
        assert len(result) == 20
        for layer in ["layer_0", "layer_1"]:
            for factor in ["0", "1"]:
                group_count = len(result[(result["layer"] == layer) & (result["factor"] == factor)])
                assert group_count == 5

    def test_sampling_ignores_missing_facet_columns(self):
        """Test that non-existent facet columns are ignored."""
        df = pd.DataFrame({"a": range(100), "value": range(100)})
        config = SamplingConfig(max_points=20, seed=42)
        # facet_columns includes "factor" which doesn't exist
        result = _apply_sampling(df, config, facet_columns=["factor", "layer"])
        # Should sample globally since no facet columns exist
        assert len(result) == 20

    def test_sampling_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        config = SamplingConfig(max_points=20, seed=42)

        result1 = _apply_sampling(df, config, facet_columns=[])
        result2 = _apply_sampling(df, config, facet_columns=[])

        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))

    def test_sampling_none_max_points_returns_original(self):
        """Test that None max_points returns DataFrame unchanged."""
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        config = SamplingConfig(max_points=None)
        result = _apply_sampling(df, config, facet_columns=[])
        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize("max_points", [-1, 0])
    def test_sampling_config_validation(self, max_points):
        """Test that invalid max_points raises error."""
        with pytest.raises(ConfigValidationError, match="positive integer"):
            SamplingConfig(max_points=max_points)
