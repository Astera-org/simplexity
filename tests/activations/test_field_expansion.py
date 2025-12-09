"""Tests for field expansion and pattern parsing in activation visualizations."""

import numpy as np
import pytest

from simplexity.activations.activation_visualizations import (
    _expand_belief_factor_mapping,
    _expand_field_mapping,
    _expand_preprocessing_fields,
    _expand_projection_key_pattern,
    _extract_base_column_name,
    _get_component_count,
    _has_field_pattern,
    _has_key_pattern,
    _parse_component_spec,
    _resolve_belief_states,
)
from simplexity.activations.visualization_configs import (
    ActivationVisualizationDataMapping,
    ActivationVisualizationFieldRef,
    CombinedMappingSection,
)
from simplexity.exceptions import ConfigValidationError


class TestPatternParsing:
    """Test pattern detection and parsing."""

    def test_parse_wildcard(self):
        """Test parsing of wildcard component pattern."""
        spec_type, start, end = _parse_component_spec("*")
        assert spec_type == "wildcard"
        assert start is None
        assert end is None

    def test_parse_range(self):
        """Test parsing of range component pattern."""
        spec_type, start, end = _parse_component_spec("0...10")
        assert spec_type == "range"
        assert start == 0
        assert end == 10

    def test_parse_range_non_zero_start(self):
        """Test parsing of range component pattern with non-zero start."""
        spec_type, start, end = _parse_component_spec("5...20")
        assert spec_type == "range"
        assert start == 5
        assert end == 20

    def test_parse_single_component(self):
        """Test parsing of single component pattern."""
        spec_type, start, end = _parse_component_spec(5)
        assert spec_type == "single"
        assert start == 5
        assert end is None

    def test_parse_none(self):
        """Test parsing of None component pattern."""
        spec_type, start, end = _parse_component_spec(None)
        assert spec_type == "none"
        assert start is None
        assert end is None

    def test_parse_invalid_range_wrong_order(self):
        """Test parsing of invalid range with start greater than end."""
        with pytest.raises(ConfigValidationError, match="start must be < end"):
            _parse_component_spec("10...5")

    def test_parse_invalid_range_equal(self):
        """Test parsing of invalid range with start equal to end."""
        with pytest.raises(ConfigValidationError, match="start must be < end"):
            _parse_component_spec("5...5")

    def test_parse_invalid_range_format(self):
        """Test parsing of invalid range format."""
        with pytest.raises(ConfigValidationError, match="Unrecognized component pattern"):
            _parse_component_spec("0..10")

    def test_parse_invalid_range_single_value(self):
        """Test parsing of invalid range with single value."""
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("10...")

    def test_parse_invalid_range_non_numeric(self):
        """Test parsing of invalid range with non-numeric values."""
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("a...b")

    def test_parse_invalid_pattern(self):
        """Test parsing of completely invalid component pattern."""
        with pytest.raises(ConfigValidationError, match="Unrecognized component pattern"):
            _parse_component_spec("invalid")

    def test_is_expansion_pattern_star(self):
        """Test detection of wildcard expansion patterns."""
        assert _has_field_pattern("prob_*")
        assert _has_field_pattern("*_prob")
        assert _has_field_pattern("prob_*_normalized")

    def test_is_expansion_pattern_range(self):
        """Test detection of range expansion patterns."""
        assert _has_field_pattern("prob_0...10")
        assert _has_field_pattern("pc_5...20")

    def test_is_expansion_pattern_no_pattern(self):
        """Test detection of non-expansion patterns."""
        assert not _has_field_pattern("prob_0")
        assert not _has_field_pattern("probability")
        assert not _has_field_pattern("pc_component")

    def test_is_expansion_pattern_multiple_patterns(self):
        """Test detection of invalid multiple patterns in field name."""
        with pytest.raises(ConfigValidationError, match="multiple patterns"):
            _has_field_pattern("prob_*_layer_*")

        with pytest.raises(ConfigValidationError, match="multiple patterns"):
            _has_field_pattern("prob_*_0...5")


class TestComponentCount:
    """Test component count determination."""

    def test_get_component_count_projections_2d(self):
        """Test getting component count from 2D projections."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(100, 10)}
        count = _get_component_count(ref, "layer_0", projections, None, False)
        assert count == 10

    def test_get_component_count_projections_different_sizes(self):
        """Test getting component count from 2D projections with different sizes."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(50, 15)}
        count = _get_component_count(ref, "layer_0", projections, None, False)
        assert count == 15

    def test_get_component_count_projections_concat_layers(self):
        """Test getting component count from concatenated layer projections."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"pca": np.random.randn(200, 20)}
        count = _get_component_count(ref, "any_layer", projections, None, True)
        assert count == 20

    def test_get_component_count_projections_1d_raises(self):
        """Test that 1D projections raise an error when getting component count."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(100)}
        with pytest.raises(ConfigValidationError, match="1D projection"):
            _get_component_count(ref, "layer_0", projections, None, False)

    def test_get_component_count_projections_3d_raises(self):
        """Test that 3D projections raise an error when getting component count."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(10, 10, 10)}
        with pytest.raises(ConfigValidationError, match="1D or 2D"):
            _get_component_count(ref, "layer_0", projections, None, False)

    def test_get_component_count_belief_states(self):
        """Test getting component count from belief states."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(100, 3)
        count = _get_component_count(ref, "layer_0", {}, belief_states, False)
        assert count == 3

    def test_get_component_count_belief_states_different_size(self):
        """Test getting component count from belief states with different size."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(50, 7)
        count = _get_component_count(ref, "layer_0", {}, belief_states, False)
        assert count == 7

    def test_get_component_count_belief_states_none_raises(self):
        """Test that None belief states raise an error when getting component count."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="not available"):
            _get_component_count(ref, "layer_0", {}, None, False)

    def test_get_component_count_belief_states_1d_raises(self):
        """Test that 1D belief states raise an error when getting component count."""
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(100)
        with pytest.raises(ConfigValidationError, match="2D"):
            _get_component_count(ref, "layer_0", {}, belief_states, False)

    def test_get_component_count_unsupported_source(self):
        """Test that unsupported sources raise an error when getting component count."""
        ref = ActivationVisualizationFieldRef(source="scalars", key="some_scalar")
        with pytest.raises(ConfigValidationError, match="not supported"):
            _get_component_count(ref, "layer_0", {}, None, False)


class TestFieldExpansion:
    """Test field mapping expansion."""

    def test_wildcard_expansion_projections(self):
        """Test detection of wildcard expansion patterns."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"layer_0_pca": np.random.randn(50, 3)}

        expanded = _expand_field_mapping("pc_*", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 3
        assert "pc_0" in expanded
        assert "pc_1" in expanded
        assert "pc_2" in expanded
        assert expanded["pc_0"].component == 0
        assert expanded["pc_1"].component == 1
        assert expanded["pc_2"].component == 2
        assert all(r.key == "pca" for r in expanded.values())
        assert all(r.source == "projections" for r in expanded.values())

    def test_wildcard_expansion_belief_states(self):
        """Test detection of wildcard expansion patterns."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component="*")
        belief_states = np.random.randn(50, 4)

        expanded = _expand_field_mapping("belief_*", ref, "layer_0", {}, {}, belief_states, False)

        assert len(expanded) == 4
        assert "belief_0" in expanded
        assert "belief_3" in expanded
        assert expanded["belief_0"].component == 0
        assert expanded["belief_3"].component == 3
        assert all(r.source == "belief_states" for r in expanded.values())

    def test_range_expansion(self):
        """Test detection of range expansion patterns."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="0...5")
        projections = {"layer_0_pca": np.random.randn(50, 10)}

        expanded = _expand_field_mapping("pc_0...5", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 5
        assert "pc_0" in expanded
        assert "pc_4" in expanded
        assert "pc_5" not in expanded
        assert expanded["pc_0"].component == 0
        assert expanded["pc_4"].component == 4

    def test_range_expansion_with_offset(self):
        """Test detection of range expansion patterns with offset."""
        ref = ActivationVisualizationFieldRef(source="projections", key="projected", component="2...5")
        projections = {"layer_0_projected": np.random.randn(50, 10)}

        expanded = _expand_field_mapping("prob_2...5", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 3
        assert "prob_2" in expanded
        assert "prob_3" in expanded
        assert "prob_4" in expanded
        assert "prob_5" not in expanded
        assert expanded["prob_2"].component == 2
        assert expanded["prob_4"].component == 4

    def test_wildcard_in_middle_of_name(self):
        """Test detection of wildcard expansion patterns."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"layer_0_pca": np.random.randn(50, 3)}

        expanded = _expand_field_mapping("component_*_normalized", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 3
        assert "component_0_normalized" in expanded
        assert "component_1_normalized" in expanded
        assert "component_2_normalized" in expanded

    def test_no_expansion_needed(self):
        """Test that no expansion occurs when component is a specific integer."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component=0)
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        expanded = _expand_field_mapping("pc_0", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 1
        assert "pc_0" in expanded
        assert expanded["pc_0"].component == 0

    def test_no_expansion_none_component(self):
        """Test that no expansion occurs when component is None."""
        ref = ActivationVisualizationFieldRef(source="metadata", key="step")
        projections = {}

        expanded = _expand_field_mapping("step", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 1
        assert "step" in expanded
        assert expanded["step"].component is None

    def test_field_pattern_without_component_pattern_raises(self):
        """Test that a field pattern without a component pattern raises an error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component=0)
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        with pytest.raises(ConfigValidationError, match="has pattern but component is not"):
            _expand_field_mapping("pc_*", ref, "layer_0", projections, {}, None, False)

    def test_component_pattern_without_field_pattern_raises(self):
        """Test that a component pattern without a field pattern raises an error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("pc_0", ref, "layer_0", projections, {}, None, False)

    def test_range_exceeds_available_components(self):
        """Test that a range exceeding available components raises an error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="0...20")
        projections = {"layer_0_pca": np.random.randn(50, 10)}

        with pytest.raises(ConfigValidationError, match="exceeds available components"):
            _expand_field_mapping("pc_0...20", ref, "layer_0", projections, {}, None, False)

    def test_range_partially_exceeds_available_components(self):
        """Test that a range partially exceeding available components raises an error."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="5...15")
        projections = {"layer_0_pca": np.random.randn(50, 10)}

        with pytest.raises(ConfigValidationError, match="exceeds available components"):
            _expand_field_mapping("pc_5...15", ref, "layer_0", projections, {}, None, False)

    def test_expansion_preserves_reducer(self):
        """Test that expansion preserves the reducer attribute."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component="*", reducer="l2_norm")
        belief_states = np.random.randn(50, 3)

        expanded = _expand_field_mapping("belief_*", ref, "layer_0", {}, {}, belief_states, False)

        assert all(r.reducer == "l2_norm" for r in expanded.values())

    def test_expansion_with_concat_layers(self):
        """Test expansion when projections are concatenated across layers."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"pca": np.random.randn(50, 5)}

        expanded = _expand_field_mapping("pc_*", ref, "layer_0", projections, {}, None, True)

        assert len(expanded) == 5
        assert all(f"pc_{i}" in expanded for i in range(5))


class TestFieldRefValidation:
    """Test ActivationVisualizationFieldRef validation."""

    def test_valid_wildcard_projections(self):
        """Test that wildcard patterns in projections are valid."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        assert ref.component == "*"

    def test_valid_range_projections(self):
        """Test that range patterns in projections are valid."""
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="0...10")
        assert ref.component == "0...10"

    def test_valid_wildcard_belief_states(self):
        """Test that wildcard patterns in belief_states are valid."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component="*")
        assert ref.component == "*"

    def test_invalid_pattern_format(self):
        """Test that invalid pattern formats raise a ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="invalid"):
            ActivationVisualizationFieldRef(source="projections", key="pca", component="invalid_pattern")

    def test_invalid_range_wrong_separator(self):
        """Test that invalid range separators raise a ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="invalid"):
            ActivationVisualizationFieldRef(source="projections", key="pca", component="0..10")

    def test_pattern_on_unsupported_source_scalars(self):
        """Test that pattern expansion is not supported for scalars source."""
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="scalars", key="some_scalar", component="*")

    def test_pattern_on_unsupported_source_metadata(self):
        """Test that pattern expansion is not supported for metadata source."""
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="metadata", key="step", component="*")

    def test_pattern_on_unsupported_source_weights(self):
        """Test that pattern expansion is not supported for weights source."""
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="weights", component="*")


class TestPreprocessingFieldExpansion:
    """Test wildcard expansion for preprocessing input_fields."""

    def test_wildcard_expansion(self):
        """Test that wildcard patterns in preprocessing fields are expanded correctly."""
        columns = ["belief_0", "belief_1", "belief_2", "belief_3", "step", "layer"]
        patterns = ["belief_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "belief_2", "belief_3"]

    def test_range_expansion(self):
        """Test that range patterns in preprocessing fields are expanded correctly."""
        columns = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5"]
        patterns = ["prob_0...3"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["prob_0", "prob_1", "prob_2"]

    def test_range_expansion_with_offset(self):
        """Test that range patterns with offsets in preprocessing fields are expanded correctly."""
        columns = ["pc_0", "pc_1", "pc_2", "pc_3", "pc_4", "pc_5", "pc_6"]
        patterns = ["pc_2...5"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["pc_2", "pc_3", "pc_4"]

    def test_mixed_patterns_and_literals(self):
        """Test that mixed wildcard patterns and literal fields are expanded correctly."""
        columns = ["belief_0", "belief_1", "belief_2", "prob_0", "prob_1", "step"]
        patterns = ["belief_*", "step"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "belief_2", "step"]

    def test_multiple_wildcards(self):
        """Test that multiple wildcard patterns in preprocessing fields are expanded correctly."""
        columns = ["belief_0", "belief_1", "prob_0", "prob_1", "prob_2"]
        patterns = ["belief_*", "prob_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "prob_0", "prob_1", "prob_2"]

    def test_wildcard_no_matches_raises(self):
        """Test that a wildcard pattern with no matches raises a ConfigValidationError."""
        columns = ["step", "layer", "sequence"]
        patterns = ["belief_*"]

        with pytest.raises(ConfigValidationError, match="did not match any columns"):
            _expand_preprocessing_fields(patterns, columns)

    def test_range_missing_column_raises(self):
        """Test that a range pattern with missing columns raises a ConfigValidationError."""
        columns = ["prob_0", "prob_1"]  # Missing prob_2
        patterns = ["prob_0...3"]

        with pytest.raises(ConfigValidationError, match="column not found"):
            _expand_preprocessing_fields(patterns, columns)

    def test_literal_fields_preserved(self):
        """Test that literal fields in preprocessing fields are preserved."""
        columns = ["field_a", "field_b", "field_c"]
        patterns = ["field_a", "field_c"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["field_a", "field_c"]

    def test_wildcard_sorts_numerically(self):
        """Test that wildcard patterns in preprocessing fields are sorted numerically."""
        columns = ["item_10", "item_2", "item_1", "item_20"]
        patterns = ["item_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        # Should be sorted by numeric value, not lexicographic
        assert expanded == ["item_1", "item_2", "item_10", "item_20"]

    def test_pattern_in_middle_of_name(self):
        """Test that patterns in the middle of field names are expanded correctly."""
        columns = ["component_0_norm", "component_1_norm", "component_2_norm"]
        patterns = ["component_*_norm"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["component_0_norm", "component_1_norm", "component_2_norm"]

    def test_empty_patterns_list(self):
        """Test that an empty patterns list returns an empty list."""
        columns = ["field_a", "field_b"]
        patterns = []

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert not expanded

    def test_range_pattern_in_middle(self):
        """Test that range patterns in the middle of field names are expanded correctly."""
        columns = ["feature_0_scaled", "feature_1_scaled", "feature_2_scaled"]
        patterns = ["feature_0...2_scaled"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["feature_0_scaled", "feature_1_scaled"]


class TestKeyPatternExpansion:
    """Test projection key pattern expansion (e.g., factor_*/projected)."""

    def test_has_key_pattern_wildcard(self):
        """Test that _has_key_pattern detects wildcard patterns correctly."""
        assert _has_key_pattern("factor_*/projected")
        assert _has_key_pattern("*/projected")
        assert _has_key_pattern("factor_*")

    def test_has_key_pattern_range(self):
        """Test that _has_key_pattern detects range patterns correctly."""
        assert _has_key_pattern("factor_0...3/projected")
        assert _has_key_pattern("0...5/projected")

    def test_has_key_pattern_none(self):
        """Test that _has_key_pattern returns False for non-pattern keys."""
        assert not _has_key_pattern(None)
        assert not _has_key_pattern("projected")
        assert not _has_key_pattern("factor_0/projected")

    def test_has_key_pattern_multiple_raises(self):
        """Test that _has_key_pattern raises an error for multiple patterns."""
        with pytest.raises(ConfigValidationError, match="multiple patterns"):
            _has_key_pattern("factor_*/layer_*/projected")

    def test_expand_projection_key_pattern_wildcard(self):
        """Test that _expand_projection_key_pattern expands wildcard patterns correctly."""
        projections = {
            "layer_0_factor_0/projected": np.random.randn(10, 3),
            "layer_0_factor_1/projected": np.random.randn(10, 3),
            "layer_0_factor_2/projected": np.random.randn(10, 3),
        }

        result = _expand_projection_key_pattern("factor_*/projected", "layer_0", projections, False)

        assert len(result) == 3
        assert result["0"] == "factor_0/projected"
        assert result["1"] == "factor_1/projected"
        assert result["2"] == "factor_2/projected"

    def test_expand_projection_key_pattern_range(self):
        """Test that _expand_projection_key_pattern expands range patterns correctly."""
        projections = {
            "layer_0_factor_0/projected": np.random.randn(10, 3),
            "layer_0_factor_1/projected": np.random.randn(10, 3),
            "layer_0_factor_2/projected": np.random.randn(10, 3),
        }

        result = _expand_projection_key_pattern("factor_0...2/projected", "layer_0", projections, False)

        assert len(result) == 2
        assert result["0"] == "factor_0/projected"
        assert result["1"] == "factor_1/projected"

    def test_expand_projection_key_pattern_concat_layers(self):
        """Test that _expand_projection_key_pattern works with concatenated layers."""
        projections = {
            "factor_0/projected": np.random.randn(10, 3),
            "factor_1/projected": np.random.randn(10, 3),
        }

        result = _expand_projection_key_pattern("factor_*/projected", "any_layer", projections, True)

        assert len(result) == 2
        assert result["0"] == "factor_0/projected"
        assert result["1"] == "factor_1/projected"

    def test_expand_projection_key_pattern_no_matches_raises(self):
        """Test that _expand_projection_key_pattern raises an error when no keys match."""
        projections = {"layer_0_pca": np.random.randn(10, 3)}

        with pytest.raises(ConfigValidationError, match="No projection keys found"):
            _expand_projection_key_pattern("factor_*/projected", "layer_0", projections, False)

    def test_field_mapping_with_key_pattern(self):
        """Test that field mappings with key patterns are expanded correctly."""
        ref = ActivationVisualizationFieldRef(
            source="projections",
            key="factor_*/projected",
            component=0,
            group_as="factor",
        )
        projections = {
            "layer_0_factor_0/projected": np.random.randn(10, 3),
            "layer_0_factor_1/projected": np.random.randn(10, 3),
        }

        expanded = _expand_field_mapping("factor_*_prob", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 2
        assert "factor_0_prob" in expanded
        assert "factor_1_prob" in expanded
        assert expanded["factor_0_prob"].key == "factor_0/projected"
        assert expanded["factor_1_prob"].key == "factor_1/projected"
        assert expanded["factor_0_prob"]._group_value == "0"  # pylint: disable=protected-access
        assert expanded["factor_1_prob"]._group_value == "1"  # pylint: disable=protected-access
        assert expanded["factor_0_prob"].group_as == "factor"

    def test_field_mapping_with_key_and_component_patterns(self):
        """Test that field mappings with key and component patterns are expanded correctly."""
        ref = ActivationVisualizationFieldRef(
            source="projections",
            key="factor_*/projected",
            component="*",
            group_as="factor",
        )
        projections = {
            "layer_0_factor_0/projected": np.random.randn(10, 3),
            "layer_0_factor_1/projected": np.random.randn(10, 3),
        }

        expanded = _expand_field_mapping("factor_*_prob_*", ref, "layer_0", projections, {}, None, False)

        # Cross-product: 2 factors * 3 components = 6 expanded fields
        assert len(expanded) == 6
        assert "factor_0_prob_0" in expanded
        assert "factor_0_prob_1" in expanded
        assert "factor_0_prob_2" in expanded
        assert "factor_1_prob_0" in expanded
        assert "factor_1_prob_1" in expanded
        assert "factor_1_prob_2" in expanded

        # Check that components are correct
        assert expanded["factor_0_prob_0"].component == 0
        assert expanded["factor_0_prob_1"].component == 1
        assert expanded["factor_1_prob_2"].component == 2

        # Check that keys and group values are correct
        assert expanded["factor_0_prob_0"].key == "factor_0/projected"
        assert expanded["factor_1_prob_0"].key == "factor_1/projected"
        assert expanded["factor_0_prob_0"]._group_value == "0"  # pylint: disable=protected-access
        assert expanded["factor_1_prob_0"]._group_value == "1"  # pylint: disable=protected-access

    def test_key_pattern_without_field_pattern_raises(self):
        """Test that a key pattern without a field pattern raises an error."""
        ref = ActivationVisualizationFieldRef(
            source="projections",
            key="factor_*/projected",
            component=0,
            group_as="factor",
        )
        projections = {"layer_0_factor_0/projected": np.random.randn(10, 3)}

        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("prob_0", ref, "layer_0", projections, {}, None, False)


class TestGroupAsValidation:
    """Test group_as parameter validation."""

    def test_key_pattern_requires_group_as(self):
        """Test that a key pattern requires the group_as parameter."""
        with pytest.raises(ConfigValidationError, match="requires `group_as`"):
            ActivationVisualizationFieldRef(
                source="projections",
                key="factor_*/projected",
                component=0,
            )

    def test_group_as_only_for_projections(self):
        """Test that group_as is only valid for projections source."""
        with pytest.raises(ConfigValidationError, match="only supported for projections"):
            ActivationVisualizationFieldRef(
                source="scalars",
                key="some_key",
                group_as="factor",
            )

    def test_valid_key_pattern_with_group_as(self):
        """Test that a valid key pattern with group_as is accepted."""
        ref = ActivationVisualizationFieldRef(
            source="projections",
            key="factor_*/projected",
            component=0,
            group_as="factor",
        )
        assert ref.group_as == "factor"
        assert ref.key == "factor_*/projected"

    def test_valid_key_pattern_with_list_group_as(self):
        """Test that a valid key pattern with list group_as is accepted."""
        ref = ActivationVisualizationFieldRef(
            source="projections",
            key="factor_*/projected",
            component=0,
            group_as=["factor", "layer"],
        )
        assert ref.group_as == ["factor", "layer"]


class TestExtractBaseColumnName:
    """Test base column name extraction for group expansion."""

    def test_extract_prefix_pattern(self):
        """Test that base column names are correctly extracted from prefixed patterns."""
        assert _extract_base_column_name("factor_0_prob_0", "0", None) == "prob_0"
        assert _extract_base_column_name("factor_1_prob_0", "1", None) == "prob_0"
        assert _extract_base_column_name("factor_2_belief", "2", None) == "belief"

    def test_extract_suffix_only_pattern_returns_original(self):
        """Test that base column names are unchanged when no prefix pattern is present."""
        # Columns without a _N_suffix pattern are returned unchanged
        # This ensures we don't strip meaningful parts of column names
        assert _extract_base_column_name("factor_0", "0", None) == "factor_0"
        assert _extract_base_column_name("group_1", "1", None) == "group_1"

    def test_no_pattern_match_returns_original(self):
        """Test that base column names are unchanged when no pattern match is found."""
        assert _extract_base_column_name("prob_0", "0", None) == "prob_0"
        assert _extract_base_column_name("some_column", "1", None) == "some_column"


class TestCombinedMappingSection:
    """Test CombinedMappingSection validation."""

    def test_valid_combined_section(self):
        """Test that a valid CombinedMappingSection is accepted."""
        section = CombinedMappingSection(
            label="prediction",
            mappings={
                "prob_0": ActivationVisualizationFieldRef(source="projections", key="proj", component=0),
            },
        )
        assert section.label == "prediction"
        assert len(section.mappings) == 1

    def test_empty_mappings_raises(self):
        """Test that an empty mappings dictionary raises a ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="must have at least one mapping"):
            CombinedMappingSection(label="empty", mappings={})


class TestCombinedDataMapping:
    """Test ActivationVisualizationDataMapping with combined sections."""

    def test_valid_combined_mapping(self):
        """Test that a valid ActivationVisualizationDataMapping with combined sections is accepted."""
        mapping = ActivationVisualizationDataMapping(
            combined=[
                CombinedMappingSection(
                    label="prediction",
                    mappings={"prob_0": ActivationVisualizationFieldRef(source="projections", key="proj", component=0)},
                ),
                CombinedMappingSection(
                    label="ground_truth",
                    mappings={"prob_0": ActivationVisualizationFieldRef(source="belief_states", component=0)},
                ),
            ],
            combine_as="data_type",
        )
        assert mapping.combined is not None
        assert len(mapping.combined) == 2
        assert mapping.combine_as == "data_type"

    def test_combined_without_combine_as_raises(self):
        """Test that an ActivationVisualizationDataMapping without 'combine_as' raises a ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="'combine_as' is required"):
            ActivationVisualizationDataMapping(
                combined=[
                    CombinedMappingSection(
                        label="prediction",
                        mappings={
                            "prob_0": ActivationVisualizationFieldRef(source="projections", key="proj", component=0)
                        },
                    ),
                ],
            )

    def test_combined_with_mappings_raises(self):
        """Test that an ActivationVisualizationDataMapping with both 'mappings' and 'combined' raises a ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Cannot use both"):
            ActivationVisualizationDataMapping(
                mappings={"prob_0": ActivationVisualizationFieldRef(source="projections", key="proj", component=0)},
                combined=[
                    CombinedMappingSection(
                        label="prediction",
                        mappings={
                            "prob_1": ActivationVisualizationFieldRef(source="projections", key="proj", component=1)
                        },
                    ),
                ],
                combine_as="data_type",
            )


class TestBeliefStateFactorPatterns:
    """Test belief state factor pattern expansion for 3D belief states."""

    def test_factor_field_only_for_belief_states(self):
        """Test that factor field is only supported for belief_states source."""
        with pytest.raises(ConfigValidationError, match="only supported for belief_states"):
            ActivationVisualizationFieldRef(
                source="projections",
                key="proj",
                factor=0,
            )

    def test_factor_pattern_requires_group_as(self):
        """Test that factor patterns require the group_as parameter."""
        with pytest.raises(ConfigValidationError, match="requires `group_as`"):
            ActivationVisualizationFieldRef(
                source="belief_states",
                factor="*",
                component=0,
            )

    def test_valid_factor_with_group_as(self):
        """Test that a valid factor pattern with group_as is accepted."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="*",
            component=0,
            group_as="factor",
        )
        assert ref.factor == "*"
        assert ref.group_as == "factor"

    def test_valid_single_factor(self):
        """Test that a valid single factor is accepted."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor=0,
            component=0,
        )
        assert ref.factor == 0

    def test_expand_belief_factor_mapping_wildcard(self):
        """Test expanding belief factor mapping with wildcard pattern."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="*",
            component=0,
            group_as="factor",
        )
        # 3D beliefs: (samples, factors, states)
        beliefs = np.random.randn(10, 3, 4)

        expanded = _expand_belief_factor_mapping("factor_*_prob", ref, beliefs)

        assert len(expanded) == 3
        assert "factor_0_prob" in expanded
        assert "factor_1_prob" in expanded
        assert "factor_2_prob" in expanded
        assert expanded["factor_0_prob"].factor == 0
        assert expanded["factor_1_prob"].factor == 1
        assert expanded["factor_2_prob"].factor == 2
        assert expanded["factor_0_prob"]._group_value == "0"  # pylint: disable=protected-access
        assert expanded["factor_1_prob"]._group_value == "1"  # pylint: disable=protected-access

    def test_expand_belief_factor_mapping_range(self):
        """Test expanding belief factor mapping with range pattern."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="0...2",
            component=0,
            group_as="factor",
        )
        beliefs = np.random.randn(10, 5, 4)

        expanded = _expand_belief_factor_mapping("factor_0...2_prob", ref, beliefs)

        assert len(expanded) == 2
        assert "factor_0_prob" in expanded
        assert "factor_1_prob" in expanded
        assert "factor_2_prob" not in expanded

    def test_expand_belief_factor_and_component_patterns(self):
        """Test expanding belief factor mapping with both factor and component patterns."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="*",
            component="*",
            group_as="factor",
        )
        beliefs = np.random.randn(10, 2, 3)

        expanded = _expand_belief_factor_mapping("factor_*_state_*", ref, beliefs)

        # Cross-product: 2 factors * 3 states = 6
        assert len(expanded) == 6
        assert "factor_0_state_0" in expanded
        assert "factor_0_state_1" in expanded
        assert "factor_0_state_2" in expanded
        assert "factor_1_state_0" in expanded
        assert "factor_1_state_1" in expanded
        assert "factor_1_state_2" in expanded
        assert expanded["factor_0_state_0"].factor == 0
        assert expanded["factor_0_state_0"].component == 0
        assert expanded["factor_1_state_2"].factor == 1
        assert expanded["factor_1_state_2"].component == 2

    def test_expand_belief_factor_mapping_2d_raises(self):
        """Test that expanding belief factor mapping with 2D beliefs raises an error."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="*",
            component=0,
            group_as="factor",
        )
        beliefs = np.random.randn(10, 4)  # 2D, not 3D

        with pytest.raises(ConfigValidationError, match="require 3D beliefs"):
            _expand_belief_factor_mapping("factor_*_prob", ref, beliefs)

    def test_expand_belief_factor_range_exceeds_raises(self):
        """Test that expanding belief factor mapping with out-of-bounds range raises an error."""
        ref = ActivationVisualizationFieldRef(
            source="belief_states",
            factor="0...10",
            component=0,
            group_as="factor",
        )
        beliefs = np.random.randn(10, 3, 4)

        with pytest.raises(ConfigValidationError, match="exceeds available factors"):
            _expand_belief_factor_mapping("factor_0...10_prob", ref, beliefs)


class TestResolveBeliefStates:
    """Test belief state resolution with factor dimension."""

    def test_resolve_2d_belief_states(self):
        """Test resolving 2D belief states without factor dimension."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component=1)
        beliefs = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])

        result = _resolve_belief_states(beliefs, ref)

        np.testing.assert_array_almost_equal(result, [0.2, 0.4])

    def test_resolve_3d_belief_states_with_factor(self):
        """Test resolving 3D belief states with specified factor."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=1, component=2)
        # Shape: (samples=2, factors=3, states=4)
        beliefs = np.random.randn(2, 3, 4)

        result = _resolve_belief_states(beliefs, ref)

        # Should select factor 1, component 2
        np.testing.assert_array_almost_equal(result, beliefs[:, 1, 2])

    def test_resolve_3d_without_factor_raises(self):
        """Test resolving 3D belief states without specifying factor raises an error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", component=0)
        beliefs = np.random.randn(10, 3, 4)

        with pytest.raises(ConfigValidationError, match="no `factor` was specified"):
            _resolve_belief_states(beliefs, ref)

    def test_resolve_2d_with_factor_raises(self):
        """Test resolving 2D belief states with factor specified raises an error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0, component=0)
        beliefs = np.random.randn(10, 4)

        with pytest.raises(ConfigValidationError, match="2D but `factor=0` was specified"):
            _resolve_belief_states(beliefs, ref)

    def test_resolve_3d_factor_out_of_bounds_raises(self):
        """Test resolving 3D belief states with out-of-bounds factor raises an error."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=5, component=0)
        beliefs = np.random.randn(10, 3, 4)

        with pytest.raises(ConfigValidationError, match="out of bounds"):
            _resolve_belief_states(beliefs, ref)

    def test_resolve_3d_with_reducer_argmax(self):
        """Test resolving 3D belief states with argmax reducer."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0, reducer="argmax")
        beliefs = np.array([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]])

        result = _resolve_belief_states(beliefs, ref)

        # Factor 0: [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]] -> argmax = [2, 0]
        np.testing.assert_array_equal(result, [2, 0])

    def test_resolve_3d_with_reducer_l2_norm(self):
        """Test resolving 3D belief states with l2_norm reducer."""
        ref = ActivationVisualizationFieldRef(source="belief_states", factor=0, reducer="l2_norm")
        beliefs = np.array([[[3.0, 4.0, 0.0]], [[1.0, 0.0, 0.0]]])

        result = _resolve_belief_states(beliefs, ref)

        np.testing.assert_array_almost_equal(result, [5.0, 1.0])
