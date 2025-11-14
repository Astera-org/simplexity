"""Tests for analysis utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.utils.analysis_utils import (
    make_prefix_groups,
    dedup_tensor_first,
    dedup_probs_sum,
    build_prefix_dataset,
    PrefixDataset,
)


class TestMakePrefixGroups:
    def test_simple_sequence(self):
        """Test prefix grouping on a simple sequence."""
        # Inputs: [[0, 1, 2]]
        inputs = jnp.array([[0, 1, 2]])
        prefix_groups = make_prefix_groups(inputs)

        # Should have 3 prefixes: (0,), (0,1), (0,1,2)
        assert len(prefix_groups) == 3
        assert (0,) in prefix_groups
        assert (0, 1) in prefix_groups
        assert (0, 1, 2) in prefix_groups

        # Check positions
        assert prefix_groups[(0,)] == [(0, 0)]
        assert prefix_groups[(0, 1)] == [(0, 1)]
        assert prefix_groups[(0, 1, 2)] == [(0, 2)]

    def test_duplicate_prefixes(self):
        """Test that duplicate prefixes are grouped together."""
        # Two sequences with overlapping prefixes
        inputs = jnp.array([[0, 1], [0, 2]])
        prefix_groups = make_prefix_groups(inputs)

        # Prefix (0,) appears twice
        assert len(prefix_groups[(0,)]) == 2
        assert (0, 0) in prefix_groups[(0,)]
        assert (1, 0) in prefix_groups[(0,)]

        # Prefixes (0,1) and (0,2) appear once each
        assert len(prefix_groups[(0, 1)]) == 1
        assert len(prefix_groups[(0, 2)]) == 1

    def test_batch_processing(self):
        """Test prefix grouping on a batch."""
        inputs = jnp.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]])
        prefix_groups = make_prefix_groups(inputs)

        # Check shared prefix (0,1)
        assert len(prefix_groups[(0, 1)]) == 2
        positions = prefix_groups[(0, 1)]
        assert (0, 1) in positions
        assert (1, 1) in positions


class TestDedupTensorFirst:
    def test_dedup_2d_tensor(self):
        """Test deduplication of 2D tensor."""
        # Create prefix groups
        inputs = jnp.array([[0, 1], [0, 1]])
        prefix_groups = make_prefix_groups(inputs)

        # Create tensor to deduplicate
        tensor = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, seq_len=2)

        dedup_values, prefixes = dedup_tensor_first(tensor, prefix_groups)

        # Should have 2 unique prefixes: (0,) and (0,1)
        assert len(prefixes) == 2
        assert (0,) in prefixes
        assert (0, 1) in prefixes

        # Values should be taken from first occurrence
        assert dedup_values.shape[0] == 2

    def test_dedup_3d_tensor(self):
        """Test deduplication of 3D tensor (e.g., activations)."""
        inputs = jnp.array([[0, 1], [0, 2]])
        prefix_groups = make_prefix_groups(inputs)

        # Create 3D tensor: (batch, seq_len, features)
        tensor = jnp.ones((2, 2, 3))
        tensor = tensor.at[0, 0, :].set(jnp.array([1.0, 2.0, 3.0]))
        tensor = tensor.at[0, 1, :].set(jnp.array([4.0, 5.0, 6.0]))
        tensor = tensor.at[1, 0, :].set(jnp.array([7.0, 8.0, 9.0]))
        tensor = tensor.at[1, 1, :].set(jnp.array([10.0, 11.0, 12.0]))

        dedup_values, prefixes = dedup_tensor_first(tensor, prefix_groups)

        # Should preserve feature dimension
        assert dedup_values.shape[1] == 3
        # Number of unique prefixes
        assert dedup_values.shape[0] == len(prefix_groups)


class TestDedupProbsSum:
    def test_sum_probabilities(self):
        """Test that probabilities are summed correctly."""
        inputs = jnp.array([[0, 1], [0, 2]])
        prefix_groups = make_prefix_groups(inputs)

        # Probabilities for each position
        probs = jnp.array([[0.3, 0.2], [0.1, 0.4]])

        dedup_probs, prefixes = dedup_probs_sum(probs, prefix_groups)

        # Find index of prefix (0,)
        idx = prefixes.index((0,))
        # Should sum probabilities from (0,0) and (1,0): 0.3 + 0.1 = 0.4
        expected_sum = 0.4
        # After normalization
        total_mass = 0.3 + 0.2 + 0.1 + 0.4  # 1.0
        expected_normalized = expected_sum / total_mass
        assert np.isclose(dedup_probs[idx], expected_normalized)

    def test_normalization(self):
        """Test that deduplicated probabilities sum to 1."""
        inputs = jnp.array([[0, 1, 2]])
        prefix_groups = make_prefix_groups(inputs)

        probs = jnp.array([[0.5, 0.3, 0.2]])

        dedup_probs, _ = dedup_probs_sum(probs, prefix_groups)

        # Should sum to 1
        assert np.isclose(dedup_probs.sum(), 1.0)

    def test_zero_probabilities(self):
        """Test handling of zero probabilities."""
        inputs = jnp.array([[0, 1]])
        prefix_groups = make_prefix_groups(inputs)

        probs = jnp.array([[0.0, 0.0]])

        dedup_probs, prefixes = dedup_probs_sum(probs, prefix_groups)

        # Should handle zeros gracefully (no division by zero)
        assert len(dedup_probs) == 2
        # All zeros means normalization should handle it


class TestBuildPrefixDataset:
    def test_build_dataset(self):
        """Test building a complete prefix dataset."""
        batch_size, seq_len = 2, 3
        n_beliefs = 3
        d_layer = 4

        inputs = jnp.array([[0, 1, 2], [0, 1, 3]])
        beliefs = jnp.ones((batch_size, seq_len, n_beliefs)) * 0.33
        probs = jnp.ones((batch_size, seq_len)) * 0.5
        activations_by_layer = {
            "layer_0": jnp.ones((batch_size, seq_len, d_layer)),
            "layer_1": jnp.ones((batch_size, seq_len, d_layer)) * 2.0,
        }

        dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)

        # Check types
        assert isinstance(dataset, PrefixDataset)
        assert isinstance(dataset.prefixes, list)
        assert len(dataset.prefixes) > 0

        # Check shapes
        n_prefixes = len(dataset.prefixes)
        assert dataset.beliefs.shape == (n_prefixes, n_beliefs)
        assert dataset.probs.shape == (n_prefixes,)
        assert dataset.activations_by_layer["layer_0"].shape == (n_prefixes, d_layer)
        assert dataset.activations_by_layer["layer_1"].shape == (n_prefixes, d_layer)

    def test_prefix_ordering_consistency(self):
        """Test that prefix ordering is consistent across all tensors."""
        inputs = jnp.array([[0, 1, 2]])
        beliefs = jnp.ones((1, 3, 2))
        probs = jnp.ones((1, 3))
        activations_by_layer = {
            "layer_0": jnp.ones((1, 3, 4)),
            "layer_1": jnp.ones((1, 3, 4)),
        }

        dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)

        # All arrays should have same length (number of unique prefixes)
        n_prefixes = len(dataset.prefixes)
        assert dataset.beliefs.shape[0] == n_prefixes
        assert dataset.probs.shape[0] == n_prefixes
        assert dataset.activations_by_layer["layer_0"].shape[0] == n_prefixes
        assert dataset.activations_by_layer["layer_1"].shape[0] == n_prefixes

    def test_empty_activations(self):
        """Test with no activation layers."""
        inputs = jnp.array([[0, 1]])
        beliefs = jnp.ones((1, 2, 2))
        probs = jnp.ones((1, 2))
        activations_by_layer = {}

        dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)

        assert len(dataset.activations_by_layer) == 0
        assert len(dataset.prefixes) > 0
