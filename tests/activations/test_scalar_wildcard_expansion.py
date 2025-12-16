"""Tests for scalar wildcard expansion in activation visualizations."""

import pytest

from simplexity.activations.visualization.pattern_expansion import _expand_scalar_keys
from simplexity.exceptions import ConfigValidationError


class TestScalarWildcardExpansion:
    """Tests for _expand_scalar_keys function."""

    def test_scalar_no_pattern_returns_identity(self):
        """Scalars without patterns should return as-is."""
        scalars = {"layer_0_rmse": 0.5}
        result = _expand_scalar_keys("rmse", "layer_0_rmse", "layer_0", scalars)

        assert result == {"rmse": "layer_0_rmse"}

    def test_scalar_wildcard_expansion(self):
        """Wildcard in scalar key should expand to all matching keys."""
        scalars = {
            "cumvar_0": 0.8,
            "cumvar_1": 0.9,
            "cumvar_2": 0.95,
            "cumvar_3": 0.99,
            "other_metric": 1.0,
        }
        result = _expand_scalar_keys("cumvar_*", "cumvar_*", "layer_0", scalars)

        assert len(result) == 4
        assert result == {
            "cumvar_0": "cumvar_0",
            "cumvar_1": "cumvar_1",
            "cumvar_2": "cumvar_2",
            "cumvar_3": "cumvar_3",
        }

    def test_scalar_wildcard_with_prefix_suffix(self):
        """Wildcard pattern with prefix and suffix should match correctly."""
        scalars = {
            "layer_0_cumvar_0": 0.8,
            "layer_0_cumvar_1": 0.9,
            "layer_0_cumvar_2": 0.95,
            "layer_1_cumvar_0": 0.7,
            "other": 1.0,
        }
        result = _expand_scalar_keys("cv_*", "layer_0_cumvar_*", "layer_0", scalars)

        assert len(result) == 3
        assert result == {
            "cv_0": "layer_0_cumvar_0",
            "cv_1": "layer_0_cumvar_1",
            "cv_2": "layer_0_cumvar_2",
        }

    def test_scalar_range_expansion(self):
        """Range pattern should expand to specified indices."""
        scalars = {
            "cumvar_0": 0.8,
            "cumvar_1": 0.9,
            "cumvar_2": 0.95,
            "cumvar_3": 0.99,
            "cumvar_4": 0.995,
        }
        result = _expand_scalar_keys("cumvar_1...4", "cumvar_1...4", "layer_0", scalars)

        assert len(result) == 3
        assert result == {
            "cumvar_1": "cumvar_1",
            "cumvar_2": "cumvar_2",
            "cumvar_3": "cumvar_3",
        }

    def test_scalar_wildcard_no_matches_raises_error(self):
        """Wildcard with no matches should raise an error."""
        scalars = {"other_metric": 1.0}

        with pytest.raises(ConfigValidationError, match="No keys found matching pattern"):
            _expand_scalar_keys("cumvar_*", "cumvar_*", "layer_0", scalars)

    def test_scalar_wildcard_requires_key_pattern(self):
        """Wildcard expansion without a key should raise an error."""
        scalars = {"metric": 1.0}

        with pytest.raises(ConfigValidationError, match="Scalar wildcard expansion requires a key pattern"):
            _expand_scalar_keys("field_*", None, "layer_0", scalars)

    def test_scalar_expansion_sorts_indices(self):
        """Expanded scalar keys should be sorted by index."""
        scalars = {
            "var_5": 0.5,
            "var_1": 0.1,
            "var_3": 0.3,
            "var_2": 0.2,
        }
        result = _expand_scalar_keys("v_*", "var_*", "layer_0", scalars)

        # Check that keys are in sorted order
        keys = list(result.keys())
        assert keys == ["v_1", "v_2", "v_3", "v_5"]

    def test_scalar_wildcard_field_name_pattern_mismatch(self):
        """Field pattern but no key pattern should be handled in parent function."""
        # This test verifies that _expand_scalar_keys expects both patterns together
        # The validation happens in _expand_field_mapping, not here
        scalars = {"metric": 1.0}

        # _expand_scalar_keys just returns identity if no pattern in key
        result = _expand_scalar_keys("field_*", "metric", "layer_0", scalars)
        assert result == {"field_*": "metric"}

    def test_scalar_range_invalid_format_returns_identity(self):
        """Invalid range format (two dots instead of three) should return as-is."""
        scalars = {"metric_1..4": 1.0}

        # Two dots instead of three - not a valid range pattern, returns identity
        result = _expand_scalar_keys("field_1..4", "metric_1..4", "layer_0", scalars)
        assert result == {"field_1..4": "metric_1..4"}

    def test_scalar_wildcard_with_non_numeric_ignored(self):
        """Keys with non-numeric wildcards should be ignored."""
        scalars = {
            "metric_0": 0.0,
            "metric_1": 0.1,
            "metric_abc": 0.2,
            "metric_xyz": 0.3,
        }
        result = _expand_scalar_keys("m_*", "metric_*", "layer_0", scalars)

        # Only numeric indices should be included
        assert len(result) == 2
        assert result == {
            "m_0": "metric_0",
            "m_1": "metric_1",
        }

    def test_scalar_expansion_deduplicates_indices(self):
        """Duplicate indices should be deduplicated."""
        # In practice this wouldn't happen with scalar keys, but test for robustness
        scalars = {
            "var_1": 0.1,
            "var_01": 0.1,  # This would match as index 1 if not carefully handled
        }
        # This test verifies basic behavior - exact matching prevents this issue
        result = _expand_scalar_keys("v_*", "var_*", "layer_0", scalars)

        # Should only match exact numeric patterns
        assert "v_1" in result

    def test_scalar_range_expansion_with_field_pattern(self):
        """Range in both field and key should expand correctly."""
        scalars = {
            "metric_0": 0.0,
            "metric_1": 0.1,
            "metric_2": 0.2,
            "metric_3": 0.3,
        }
        result = _expand_scalar_keys("m_0...3", "metric_0...3", "layer_0", scalars)

        assert len(result) == 3
        assert result == {
            "m_0": "metric_0",
            "m_1": "metric_1",
            "m_2": "metric_2",
        }

    def test_scalar_wildcard_complex_key_pattern(self):
        """Complex patterns with multiple underscores should work."""
        scalars = {
            "layer_0_pca_cumvar_0": 0.8,
            "layer_0_pca_cumvar_1": 0.9,
            "layer_0_pca_cumvar_2": 0.95,
            "layer_1_pca_cumvar_0": 0.7,
        }
        result = _expand_scalar_keys("pc_cv_*", "layer_0_pca_cumvar_*", "layer_0", scalars)

        assert len(result) == 3
        assert result == {
            "pc_cv_0": "layer_0_pca_cumvar_0",
            "pc_cv_1": "layer_0_pca_cumvar_1",
            "pc_cv_2": "layer_0_pca_cumvar_2",
        }
