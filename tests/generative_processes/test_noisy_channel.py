"""Tests for noisy channel functionality."""

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.noisy_channel import (
    apply_noisy_channel,
    compute_joint_blur_matrix,
)


class TestApplyNoisyChannel:
    """Tests for apply_noisy_channel function."""

    def test_zero_epsilon_returns_unchanged(self):
        """Test that zero noise epsilon returns the original matrices."""
        matrices = jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.7], [0.6, 0.4]]])
        result = apply_noisy_channel(matrices, noise_epsilon=0.0)
        chex.assert_trees_all_close(result, matrices)

    def test_one_epsilon_returns_uniform_blur(self):
        """Test that noise epsilon of 1.0 produces uniform blur."""
        matrices = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        result = apply_noisy_channel(matrices, noise_epsilon=1.0)
        chex.assert_trees_all_close(result[0], result[1])

    def test_intermediate_epsilon(self):
        """Test intermediate noise epsilon values produce expected blur."""
        matrices = jnp.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        result = apply_noisy_channel(matrices, noise_epsilon=0.5)
        expected = jnp.array([[[0.75, 0.25]], [[0.25, 0.75]]])
        chex.assert_trees_all_close(result, expected)

    def test_invalid_epsilon_negative_raises(self):
        """Test that negative noise epsilon raises ValueError."""
        matrices = jnp.array([[[1.0]]])
        with pytest.raises(ValueError, match="noise_epsilon must be in"):
            apply_noisy_channel(matrices, noise_epsilon=-0.1)

    def test_invalid_epsilon_greater_than_one_raises(self):
        """Test that noise epsilon greater than 1 raises ValueError."""
        matrices = jnp.array([[[1.0]]])
        with pytest.raises(ValueError, match="noise_epsilon must be in"):
            apply_noisy_channel(matrices, noise_epsilon=1.1)

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        matrices = jnp.ones((3, 4, 4))
        result = apply_noisy_channel(matrices, noise_epsilon=0.2)
        assert result.shape == matrices.shape


class TestComputeJointBlurMatrix:
    """Tests for compute_joint_blur_matrix function."""

    def test_shape_single_factor(self):
        """Test blur matrix shape for single factor."""
        blur = compute_joint_blur_matrix((4,), noise_epsilon=0.1)
        assert blur.shape == (4, 4)

    def test_shape_two_factors(self):
        """Test blur matrix shape for two factors."""
        blur = compute_joint_blur_matrix((2, 3), noise_epsilon=0.1)
        assert blur.shape == (6, 6)

    def test_shape_three_factors(self):
        """Test blur matrix shape for three factors."""
        blur = compute_joint_blur_matrix((2, 3, 4), noise_epsilon=0.2)
        assert blur.shape == (24, 24)

    def test_row_stochastic(self):
        """Test that blur matrix is row stochastic."""
        blur = compute_joint_blur_matrix((2, 3), noise_epsilon=0.2)
        row_sums = jnp.sum(blur, axis=1)
        chex.assert_trees_all_close(row_sums, jnp.ones(6))

    def test_zero_epsilon_is_identity(self):
        """Test that zero noise epsilon produces identity matrix."""
        blur = compute_joint_blur_matrix((2, 3), noise_epsilon=0.0)
        chex.assert_trees_all_close(blur, jnp.eye(6))

    def test_one_epsilon_is_uniform(self):
        """Test that noise epsilon of 1.0 produces uniform matrix."""
        blur = compute_joint_blur_matrix((2, 3), noise_epsilon=1.0)
        expected = jnp.ones((6, 6)) / 6
        chex.assert_trees_all_close(blur, expected)

    def test_invalid_epsilon_raises(self):
        """Test that invalid noise epsilon values raise ValueError."""
        with pytest.raises(ValueError, match="noise_epsilon must be in"):
            compute_joint_blur_matrix((2, 3), noise_epsilon=-0.1)
        with pytest.raises(ValueError, match="noise_epsilon must be in"):
            compute_joint_blur_matrix((2, 3), noise_epsilon=1.5)
