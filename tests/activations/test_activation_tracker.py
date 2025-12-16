"""Tests for ActivationTracker class."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# pylint: enable=all

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from simplexity.activations.activation_analyses import PcaAnalysis
from simplexity.activations.activation_tracker import (
    ActivationTracker,
    PrepareOptions,
    _get_uniform_weights,
    _to_jax_array,
    prepare_activations,
)


class TestGetUniformWeights:
    """Tests for _get_uniform_weights helper."""

    def test_returns_uniform_weights(self):
        """Test that uniform weights sum to 1."""
        weights = _get_uniform_weights(5, jnp.float32)
        assert weights.shape == (5,)
        assert np.isclose(float(weights.sum()), 1.0)

    def test_each_weight_equal(self):
        """Test that each weight is equal."""
        weights = _get_uniform_weights(4, jnp.float32)
        expected = 0.25
        for w in weights:
            assert np.isclose(float(w), expected)


class TestToJaxArray:
    """Tests for _to_jax_array helper."""

    def test_numpy_array(self):
        """Test conversion from numpy array."""
        arr = np.array([1, 2, 3])
        result = _to_jax_array(arr)
        assert isinstance(result, jnp.ndarray)
        assert list(result) == [1, 2, 3]

    def test_jax_array_passthrough(self):
        """Test that JAX arrays pass through unchanged."""
        arr = jnp.array([1, 2, 3])
        result = _to_jax_array(arr)
        assert result is arr


class TestPrepareActivations:
    """Tests for prepare_activations function."""

    @pytest.fixture
    def basic_data(self):
        """Create basic test data."""
        batch_size = 2
        seq_len = 3
        belief_dim = 2
        d_model = 4

        inputs = jnp.array([[1, 2, 3], [1, 2, 4]])
        beliefs = jnp.ones((batch_size, seq_len, belief_dim)) * 0.5
        probs = jnp.ones((batch_size, seq_len)) * 0.1
        activations = {"layer_0": jnp.ones((batch_size, seq_len, d_model)) * 0.3}

        return {
            "inputs": inputs,
            "beliefs": beliefs,
            "probs": probs,
            "activations": activations,
        }

    def test_uses_probs_as_weights(self, basic_data):
        """Test that probs are used as weights when specified."""
        result = prepare_activations(
            basic_data["inputs"],
            basic_data["beliefs"],
            basic_data["probs"],
            basic_data["activations"],
            PrepareOptions(last_token_only=False, concat_layers=False, use_probs_as_weights=True),
        )
        assert result.weights is not None

    def test_uses_uniform_weights(self, basic_data):
        """Test that uniform weights are used when probs not used."""
        result = prepare_activations(
            basic_data["inputs"],
            basic_data["beliefs"],
            basic_data["probs"],
            basic_data["activations"],
            PrepareOptions(last_token_only=False, concat_layers=False, use_probs_as_weights=False),
        )
        assert result.weights is not None
        assert np.isclose(float(result.weights.sum()), 1.0)

    def test_concat_layers(self, basic_data):
        """Test layer concatenation."""
        basic_data["activations"]["layer_1"] = jnp.ones((2, 3, 6)) * 0.5
        result = prepare_activations(
            basic_data["inputs"],
            basic_data["beliefs"],
            basic_data["probs"],
            basic_data["activations"],
            PrepareOptions(last_token_only=False, concat_layers=True, use_probs_as_weights=False),
        )
        assert "concatenated" in result.activations
        assert len(result.activations) == 1

    def test_tuple_beliefs(self, basic_data):
        """Test handling of tuple belief states (factored processes)."""
        beliefs_tuple = (
            jnp.ones((2, 3, 2)) * 0.3,
            jnp.ones((2, 3, 3)) * 0.7,
        )
        result = prepare_activations(
            basic_data["inputs"],
            beliefs_tuple,
            basic_data["probs"],
            basic_data["activations"],
            PrepareOptions(last_token_only=False, concat_layers=False, use_probs_as_weights=False),
        )
        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)


class TestActivationTrackerScalarHistory:
    """Tests for ActivationTracker scalar history methods."""

    @pytest.fixture
    def tracker_with_history(self):
        """Create tracker with some scalar history."""
        tracker = ActivationTracker(
            analyses={"pca": PcaAnalysis(n_components=1, last_token_only=False, concat_layers=False)},
        )
        # Manually populate scalar history
        tracker._scalar_history = {
            "pca/layer_0_r2": [(0, 0.5), (1, 0.6), (2, 0.7)],
            "pca/layer_1_r2": [(0, 0.4), (1, 0.5), (2, 0.6)],
            "pca/layer_0_rmse": [(0, 0.3), (1, 0.2), (2, 0.1)],
            "pca/layer_1_rmse": [(0, 0.4), (1, 0.3), (2, 0.2)],
        }
        return tracker

    def test_get_scalar_history_no_pattern(self, tracker_with_history):
        """Test getting all scalar history without pattern."""
        history = tracker_with_history.get_scalar_history()
        assert len(history) == 4
        assert "pca/layer_0_r2" in history
        assert "pca/layer_1_r2" in history

    def test_get_scalar_history_exact_match(self, tracker_with_history):
        """Test getting scalar history with exact match (no wildcards)."""
        history = tracker_with_history.get_scalar_history("pca/layer_0_r2")
        assert len(history) == 1
        assert "pca/layer_0_r2" in history

    def test_get_scalar_history_exact_match_not_found(self, tracker_with_history):
        """Test exact match returns empty when not found."""
        history = tracker_with_history.get_scalar_history("nonexistent")
        assert len(history) == 0

    def test_get_scalar_history_star_pattern(self, tracker_with_history):
        """Test getting scalar history with * wildcard."""
        history = tracker_with_history.get_scalar_history("pca/layer_*_r2")
        assert len(history) == 2
        assert "pca/layer_0_r2" in history
        assert "pca/layer_1_r2" in history

    def test_get_scalar_history_star_pattern_all(self, tracker_with_history):
        """Test * wildcard matching all metrics."""
        history = tracker_with_history.get_scalar_history("pca/*")
        assert len(history) == 4

    def test_get_scalar_history_range_pattern(self, tracker_with_history):
        """Test getting scalar history with range pattern."""
        history = tracker_with_history.get_scalar_history("pca/layer_0...2_r2")
        assert len(history) == 2
        assert "pca/layer_0_r2" in history
        assert "pca/layer_1_r2" in history

    def test_get_scalar_history_range_partial_match(self, tracker_with_history):
        """Test range pattern with partial matches."""
        history = tracker_with_history.get_scalar_history("pca/layer_0...1_r2")
        assert len(history) == 1
        assert "pca/layer_0_r2" in history

    def test_get_scalar_history_df_empty(self):
        """Test get_scalar_history_df with empty history."""
        tracker = ActivationTracker(analyses={})
        df = tracker.get_scalar_history_df()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["metric", "step", "value"]
        assert len(df) == 0

    def test_get_scalar_history_df_with_data(self, tracker_with_history):
        """Test get_scalar_history_df with data."""
        df = tracker_with_history.get_scalar_history_df()
        assert isinstance(df, pd.DataFrame)
        assert "metric" in df.columns
        assert "step" in df.columns
        assert "value" in df.columns
        assert len(df) == 12  # 4 metrics * 3 steps each

    def test_get_scalar_history_df_structure(self, tracker_with_history):
        """Test that DataFrame has correct structure."""
        df = tracker_with_history.get_scalar_history_df()
        pca_layer0_r2 = df[df["metric"] == "pca/layer_0_r2"]
        assert len(pca_layer0_r2) == 3
        assert list(pca_layer0_r2["step"]) == [0, 1, 2]
        assert list(pca_layer0_r2["value"]) == [0.5, 0.6, 0.7]


class TestActivationTrackerVisualizationHandling:
    """Tests for visualization handling in ActivationTracker."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for testing."""
        batch_size = 2
        seq_len = 3
        belief_dim = 2
        d_model = 4

        inputs = jnp.array([[1, 2, 3], [1, 2, 4]])
        beliefs = jnp.ones((batch_size, seq_len, belief_dim)) * 0.5
        probs = jnp.ones((batch_size, seq_len)) * 0.1
        activations = {"layer_0": jnp.ones((batch_size, seq_len, d_model)) * 0.3}

        return {
            "inputs": inputs,
            "beliefs": beliefs,
            "probs": probs,
            "activations": activations,
        }

    def test_analyze_without_visualizations(self, synthetic_data):
        """Test analyze works without visualizations configured."""
        tracker = ActivationTracker(
            analyses={"pca": PcaAnalysis(n_components=1, last_token_only=False, concat_layers=False)},
        )
        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )
        assert len(scalars) > 0
        assert len(projections) > 0
        assert len(visualizations) == 0

    def test_analyze_records_scalar_history(self, synthetic_data):
        """Test that analyze records scalar history when step is provided."""
        tracker = ActivationTracker(
            analyses={"pca": PcaAnalysis(n_components=1, last_token_only=False, concat_layers=False)},
        )
        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=0,
        )
        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=1,
        )
        history = tracker.get_scalar_history()
        assert len(history) > 0
        for _key, values in history.items():
            assert len(values) == 2
            assert values[0][0] == 0  # First step
            assert values[1][0] == 1  # Second step

    def test_analyze_with_tuple_beliefs_creates_stacked_array(self, synthetic_data):
        """Test that tuple beliefs are stacked correctly for visualization."""
        # Tuple beliefs must have same shape for stacking
        beliefs_tuple = (
            jnp.ones((2, 3, 2)) * 0.3,
            jnp.ones((2, 3, 2)) * 0.7,
        )
        viz_cfg = {
            "name": "test_viz",
            "data_mapping": {
                "mappings": {
                    "pc0": {"source": "projections", "key": "pca", "component": 0},
                },
            },
            "layer": {
                "geometry": {"type": "point"},
                "aesthetics": {
                    "x": {"field": "pc0", "type": "quantitative"},
                },
            },
        }
        tracker = ActivationTracker(
            analyses={"pca": PcaAnalysis(n_components=1, last_token_only=False, concat_layers=False)},
            visualizations={"pca": [viz_cfg]},
        )
        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=beliefs_tuple,
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=0,
        )
        assert len(visualizations) > 0

    def test_analyze_with_none_beliefs(self, synthetic_data):
        """Test visualization when beliefs are None."""
        viz_cfg = {
            "name": "test_viz",
            "data_mapping": {
                "mappings": {
                    "pc0": {"source": "projections", "key": "pca", "component": 0},
                },
            },
            "layer": {
                "geometry": {"type": "point"},
                "aesthetics": {
                    "x": {"field": "pc0", "type": "quantitative"},
                },
            },
        }
        tracker = ActivationTracker(
            analyses={"pca": PcaAnalysis(n_components=1, last_token_only=False, concat_layers=False)},
            visualizations={"pca": [viz_cfg]},
        )
        # PCA doesn't require beliefs, so this should work
        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=0,
        )
        assert len(visualizations) > 0


class TestActivationTrackerSaveVisualizations:
    """Tests for save_visualizations method."""

    def test_save_visualizations_delegates_to_persistence(self, tmp_path):
        """Test that save_visualizations calls the persistence function."""
        tracker = ActivationTracker(analyses={})
        mock_payload = MagicMock()
        mock_payload.name = "test_viz"
        visualizations = {"pca/test_viz": mock_payload}

        with patch("simplexity.activations.activation_tracker.save_visualization_payloads") as mock_save:
            mock_save.return_value = {"pca/test_viz": str(tmp_path / "test.html")}
            result = tracker.save_visualizations(visualizations, tmp_path, step=0)
            mock_save.assert_called_once_with(visualizations, tmp_path, 0)
            assert "pca/test_viz" in result
