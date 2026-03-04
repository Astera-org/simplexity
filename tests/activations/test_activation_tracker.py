"""Tests for ActivationTracker class."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# pylint: enable=all

import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.activations.activation_tracker import (
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
