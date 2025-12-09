import numpy as np
import pytest

from simplexity.activations.activation_visualizations import (
    _expand_field_mapping,
    _expand_preprocessing_fields,
    _get_component_count,
    _has_field_pattern,
    _parse_component_spec,
)
from simplexity.activations.visualization_configs import ActivationVisualizationFieldRef
from simplexity.exceptions import ConfigValidationError


class TestPatternParsing:
    """Test pattern detection and parsing."""

    def test_parse_wildcard(self):
        spec_type, start, end = _parse_component_spec("*")
        assert spec_type == "wildcard"
        assert start is None
        assert end is None

    def test_parse_range(self):
        spec_type, start, end = _parse_component_spec("0...10")
        assert spec_type == "range"
        assert start == 0
        assert end == 10

    def test_parse_range_non_zero_start(self):
        spec_type, start, end = _parse_component_spec("5...20")
        assert spec_type == "range"
        assert start == 5
        assert end == 20

    def test_parse_single_component(self):
        spec_type, start, end = _parse_component_spec(5)
        assert spec_type == "single"
        assert start == 5
        assert end is None

    def test_parse_none(self):
        spec_type, start, end = _parse_component_spec(None)
        assert spec_type == "none"
        assert start is None
        assert end is None

    def test_parse_invalid_range_wrong_order(self):
        with pytest.raises(ConfigValidationError, match="start must be < end"):
            _parse_component_spec("10...5")

    def test_parse_invalid_range_equal(self):
        with pytest.raises(ConfigValidationError, match="start must be < end"):
            _parse_component_spec("5...5")

    def test_parse_invalid_range_format(self):
        with pytest.raises(ConfigValidationError, match="Unrecognized component pattern"):
            _parse_component_spec("0..10")

    def test_parse_invalid_range_single_value(self):
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("10...")

    def test_parse_invalid_range_non_numeric(self):
        with pytest.raises(ConfigValidationError, match="Invalid range"):
            _parse_component_spec("a...b")

    def test_parse_invalid_pattern(self):
        with pytest.raises(ConfigValidationError, match="Unrecognized component pattern"):
            _parse_component_spec("invalid")

    def test_is_expansion_pattern_star(self):
        assert _has_field_pattern("prob_*")
        assert _has_field_pattern("*_prob")
        assert _has_field_pattern("prob_*_normalized")

    def test_is_expansion_pattern_range(self):
        assert _has_field_pattern("prob_0...10")
        assert _has_field_pattern("pc_5...20")

    def test_is_expansion_pattern_no_pattern(self):
        assert not _has_field_pattern("prob_0")
        assert not _has_field_pattern("probability")
        assert not _has_field_pattern("pc_component")

    def test_is_expansion_pattern_multiple_patterns(self):
        with pytest.raises(ConfigValidationError, match="multiple patterns"):
            _has_field_pattern("prob_*_layer_*")

        with pytest.raises(ConfigValidationError, match="multiple patterns"):
            _has_field_pattern("prob_*_0...5")


class TestComponentCount:
    """Test component count determination."""

    def test_get_component_count_projections_2d(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(100, 10)}
        count = _get_component_count(ref, "layer_0", projections, None, False)
        assert count == 10

    def test_get_component_count_projections_different_sizes(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(50, 15)}
        count = _get_component_count(ref, "layer_0", projections, None, False)
        assert count == 15

    def test_get_component_count_projections_concat_layers(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"pca": np.random.randn(200, 20)}
        count = _get_component_count(ref, "any_layer", projections, None, True)
        assert count == 20

    def test_get_component_count_projections_1d_raises(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(100)}
        with pytest.raises(ConfigValidationError, match="1D projection"):
            _get_component_count(ref, "layer_0", projections, None, False)

    def test_get_component_count_projections_3d_raises(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca")
        projections = {"layer_0_pca": np.random.randn(10, 10, 10)}
        with pytest.raises(ConfigValidationError, match="1D or 2D"):
            _get_component_count(ref, "layer_0", projections, None, False)

    def test_get_component_count_belief_states(self):
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(100, 3)
        count = _get_component_count(ref, "layer_0", {}, belief_states, False)
        assert count == 3

    def test_get_component_count_belief_states_different_size(self):
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(50, 7)
        count = _get_component_count(ref, "layer_0", {}, belief_states, False)
        assert count == 7

    def test_get_component_count_belief_states_none_raises(self):
        ref = ActivationVisualizationFieldRef(source="belief_states")
        with pytest.raises(ConfigValidationError, match="not available"):
            _get_component_count(ref, "layer_0", {}, None, False)

    def test_get_component_count_belief_states_1d_raises(self):
        ref = ActivationVisualizationFieldRef(source="belief_states")
        belief_states = np.random.randn(100)
        with pytest.raises(ConfigValidationError, match="2D"):
            _get_component_count(ref, "layer_0", {}, belief_states, False)

    def test_get_component_count_unsupported_source(self):
        ref = ActivationVisualizationFieldRef(source="scalars", key="some_scalar")
        with pytest.raises(ConfigValidationError, match="not supported"):
            _get_component_count(ref, "layer_0", {}, None, False)


class TestFieldExpansion:
    """Test field mapping expansion."""

    def test_wildcard_expansion_projections(self):
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
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"layer_0_pca": np.random.randn(50, 3)}

        expanded = _expand_field_mapping("component_*_normalized", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 3
        assert "component_0_normalized" in expanded
        assert "component_1_normalized" in expanded
        assert "component_2_normalized" in expanded

    def test_no_expansion_needed(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component=0)
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        expanded = _expand_field_mapping("pc_0", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 1
        assert "pc_0" in expanded
        assert expanded["pc_0"].component == 0

    def test_no_expansion_none_component(self):
        ref = ActivationVisualizationFieldRef(source="metadata", key="step")
        projections = {}

        expanded = _expand_field_mapping("step", ref, "layer_0", projections, {}, None, False)

        assert len(expanded) == 1
        assert "step" in expanded
        assert expanded["step"].component is None

    def test_field_pattern_without_component_pattern_raises(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component=0)
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        with pytest.raises(ConfigValidationError, match="has pattern but component is not"):
            _expand_field_mapping("pc_*", ref, "layer_0", projections, {}, None, False)

    def test_component_pattern_without_field_pattern_raises(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"layer_0_pca": np.random.randn(50, 5)}

        with pytest.raises(ConfigValidationError, match="requires field name pattern"):
            _expand_field_mapping("pc_0", ref, "layer_0", projections, {}, None, False)

    def test_range_exceeds_available_components(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="0...20")
        projections = {"layer_0_pca": np.random.randn(50, 10)}

        with pytest.raises(ConfigValidationError, match="exceeds available components"):
            _expand_field_mapping("pc_0...20", ref, "layer_0", projections, {}, None, False)

    def test_range_partially_exceeds_available_components(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="5...15")
        projections = {"layer_0_pca": np.random.randn(50, 10)}

        with pytest.raises(ConfigValidationError, match="exceeds available components"):
            _expand_field_mapping("pc_5...15", ref, "layer_0", projections, {}, None, False)

    def test_expansion_preserves_reducer(self):
        ref = ActivationVisualizationFieldRef(source="belief_states", component="*", reducer="l2_norm")
        belief_states = np.random.randn(50, 3)

        expanded = _expand_field_mapping("belief_*", ref, "layer_0", {}, {}, belief_states, False)

        assert all(r.reducer == "l2_norm" for r in expanded.values())

    def test_expansion_with_concat_layers(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        projections = {"pca": np.random.randn(50, 5)}

        expanded = _expand_field_mapping("pc_*", ref, "layer_0", projections, {}, None, True)

        assert len(expanded) == 5
        assert all(f"pc_{i}" in expanded for i in range(5))


class TestFieldRefValidation:
    """Test ActivationVisualizationFieldRef validation."""

    def test_valid_wildcard_projections(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="*")
        assert ref.component == "*"

    def test_valid_range_projections(self):
        ref = ActivationVisualizationFieldRef(source="projections", key="pca", component="0...10")
        assert ref.component == "0...10"

    def test_valid_wildcard_belief_states(self):
        ref = ActivationVisualizationFieldRef(source="belief_states", component="*")
        assert ref.component == "*"

    def test_invalid_pattern_format(self):
        with pytest.raises(ConfigValidationError, match="invalid"):
            ActivationVisualizationFieldRef(source="projections", key="pca", component="invalid_pattern")

    def test_invalid_range_wrong_separator(self):
        with pytest.raises(ConfigValidationError, match="invalid"):
            ActivationVisualizationFieldRef(source="projections", key="pca", component="0..10")

    def test_pattern_on_unsupported_source_scalars(self):
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="scalars", key="some_scalar", component="*")

    def test_pattern_on_unsupported_source_metadata(self):
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="metadata", key="step", component="*")

    def test_pattern_on_unsupported_source_weights(self):
        with pytest.raises(ConfigValidationError, match="only supported for projections/belief_states"):
            ActivationVisualizationFieldRef(source="weights", component="*")


class TestPreprocessingFieldExpansion:
    """Test wildcard expansion for preprocessing input_fields."""

    def test_wildcard_expansion(self):
        columns = ["belief_0", "belief_1", "belief_2", "belief_3", "step", "layer"]
        patterns = ["belief_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "belief_2", "belief_3"]

    def test_range_expansion(self):
        columns = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5"]
        patterns = ["prob_0...3"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["prob_0", "prob_1", "prob_2"]

    def test_range_expansion_with_offset(self):
        columns = ["pc_0", "pc_1", "pc_2", "pc_3", "pc_4", "pc_5", "pc_6"]
        patterns = ["pc_2...5"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["pc_2", "pc_3", "pc_4"]

    def test_mixed_patterns_and_literals(self):
        columns = ["belief_0", "belief_1", "belief_2", "prob_0", "prob_1", "step"]
        patterns = ["belief_*", "step"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "belief_2", "step"]

    def test_multiple_wildcards(self):
        columns = ["belief_0", "belief_1", "prob_0", "prob_1", "prob_2"]
        patterns = ["belief_*", "prob_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["belief_0", "belief_1", "prob_0", "prob_1", "prob_2"]

    def test_wildcard_no_matches_raises(self):
        columns = ["step", "layer", "sequence"]
        patterns = ["belief_*"]

        with pytest.raises(ConfigValidationError, match="did not match any columns"):
            _expand_preprocessing_fields(patterns, columns)

    def test_range_missing_column_raises(self):
        columns = ["prob_0", "prob_1"]  # Missing prob_2
        patterns = ["prob_0...3"]

        with pytest.raises(ConfigValidationError, match="column not found"):
            _expand_preprocessing_fields(patterns, columns)

    def test_literal_fields_preserved(self):
        columns = ["field_a", "field_b", "field_c"]
        patterns = ["field_a", "field_c"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["field_a", "field_c"]

    def test_wildcard_sorts_numerically(self):
        columns = ["item_10", "item_2", "item_1", "item_20"]
        patterns = ["item_*"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        # Should be sorted by numeric value, not lexicographic
        assert expanded == ["item_1", "item_2", "item_10", "item_20"]

    def test_pattern_in_middle_of_name(self):
        columns = ["component_0_norm", "component_1_norm", "component_2_norm"]
        patterns = ["component_*_norm"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["component_0_norm", "component_1_norm", "component_2_norm"]

    def test_empty_patterns_list(self):
        columns = ["field_a", "field_b"]
        patterns = []

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == []

    def test_range_pattern_in_middle(self):
        columns = ["feature_0_scaled", "feature_1_scaled", "feature_2_scaled"]
        patterns = ["feature_0...2_scaled"]

        expanded = _expand_preprocessing_fields(patterns, columns)

        assert expanded == ["feature_0_scaled", "feature_1_scaled"]
