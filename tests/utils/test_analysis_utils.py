"""Tests for analysis utilities."""

import jax
import jax.numpy as jnp
import pytest

from simplexity.utils.analysis_utils import (
    build_last_token_dataset,
    build_prefix_dataset,
    dedup_last_token_probs_sum,
    dedup_last_token_tensor_first,
    dedup_probs_sum,
    dedup_tensor_first,
    make_prefix_groups,
    make_sequence_groups,
)


@pytest.fixture
def simple_inputs():
    """Create simple test inputs with known duplicates."""
    return jnp.array(
        [
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
        ]
    )


@pytest.fixture
def simple_beliefs():
    """Create simple belief states."""
    return jnp.array(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        ]
    )


@pytest.fixture
def simple_probs():
    """Create simple probabilities."""
    return jnp.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
            [0.3, 0.2, 0.5],
        ]
    )


@pytest.fixture
def simple_activations():
    """Create simple activations."""
    return {
        "layer_0": jnp.ones((3, 3, 4)) * 0.1,
        "layer_1": jnp.ones((3, 3, 6)) * 0.2,
    }


class TestMakePrefixGroups:
    """Test make_prefix_groups function."""

    def test_unique_prefixes(self):
        """Test with all unique prefixes."""
        inputs = jnp.array([[1, 2, 3], [4, 5, 6]])
        groups = make_prefix_groups(inputs)

        # Should have 6 unique prefixes total (3 per sequence)
        assert len(groups) == 6
        # Each prefix should appear exactly once
        for positions in groups.values():
            assert len(positions) == 1

    def test_duplicate_prefixes(self, simple_inputs):
        """Test with duplicate prefixes across sequences."""
        groups = make_prefix_groups(simple_inputs)

        # Prefix [1] appears 3 times (once per sequence)
        assert len(groups[(1,)]) == 3

        # Prefix [1, 2] appears 3 times
        assert len(groups[(1, 2)]) == 3

        # Prefix [1, 2, 3] appears 2 times (sequences 0 and 2)
        assert len(groups[(1, 2, 3)]) == 2

        # Prefix [1, 2, 4] appears 1 time (sequence 1)
        assert len(groups[(1, 2, 4)]) == 1

    def test_positions_correct(self, simple_inputs):
        """Test that positions are correctly recorded."""
        groups = make_prefix_groups(simple_inputs)

        # Check positions for prefix [1, 2]
        positions = groups[(1, 2)]
        assert (0, 1) in positions  # sequence 0, position 1
        assert (1, 1) in positions  # sequence 1, position 1
        assert (2, 1) in positions  # sequence 2, position 1

    def test_single_token_sequences(self):
        """Test with single-token sequences."""
        inputs = jnp.array([[1], [2], [1]])
        groups = make_prefix_groups(inputs)

        assert len(groups) == 2  # Two unique prefixes: [1] and [2]
        assert len(groups[(1,)]) == 2  # [1] appears twice
        assert len(groups[(2,)]) == 1  # [2] appears once


class TestMakeSequenceGroups:
    """Test make_sequence_groups function."""

    def test_unique_sequences(self):
        """Test with all unique sequences."""
        inputs = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        groups = make_sequence_groups(inputs)

        assert len(groups) == 3
        for indices in groups.values():
            assert len(indices) == 1

    def test_duplicate_sequences(self, simple_inputs):
        """Test with duplicate sequences."""
        groups = make_sequence_groups(simple_inputs)

        # Should have 2 unique sequences
        assert len(groups) == 2

        # Sequence [1, 2, 3] appears at indices 0 and 2
        assert set(groups[(1, 2, 3)]) == {0, 2}

        # Sequence [1, 2, 4] appears at index 1
        assert groups[(1, 2, 4)] == [1]

    def test_all_identical_sequences(self):
        """Test when all sequences are identical."""
        inputs = jnp.array([[1, 2], [1, 2], [1, 2]])
        groups = make_sequence_groups(inputs)

        assert len(groups) == 1
        assert groups[(1, 2)] == [0, 1, 2]


class TestDedupTensorFirst:
    """Test dedup_tensor_first function."""

    def test_takes_first_occurrence(self, simple_beliefs):
        """Test that first occurrence is selected for duplicates."""
        inputs = jnp.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        groups = make_prefix_groups(inputs)

        # Get beliefs for position 2 (last position)
        last_pos_beliefs = simple_beliefs[:, 2, :]  # shape (3, 2)

        dedup_values, prefixes = dedup_tensor_first(last_pos_beliefs, groups)

        # Should have 6 unique prefixes
        assert dedup_values.shape[0] == 6
        assert len(prefixes) == 6

    def test_preserves_feature_dimension(self, simple_beliefs):
        """Test that feature dimensions are preserved."""
        inputs = jnp.array([[1, 2], [3, 4]])
        groups = make_prefix_groups(inputs)

        # Use first 2 sequences from beliefs (shape: 2, 3, 2)
        beliefs_subset = simple_beliefs[:2, :, :]

        dedup_values, _ = dedup_tensor_first(beliefs_subset, groups)

        # Should preserve last dimension (belief dimension = 2)
        assert dedup_values.shape[1] == beliefs_subset.shape[2]

    def test_correct_first_value_selected(self):
        """Test that the correct first value is selected."""
        inputs = jnp.array([[1, 2], [1, 2]])
        groups = make_prefix_groups(inputs)

        # Create tensor with distinct values (batch, seq_len, features)
        tensor = jnp.array(
            [
                [[10.0, 11.0], [12.0, 13.0]],  # sequence 0
                [[20.0, 21.0], [22.0, 23.0]],  # sequence 1
            ]
        )

        dedup_values, prefixes = dedup_tensor_first(tensor, groups)

        # Prefix [1, 2] should get value from first occurrence (sequence 0, position 1)
        prefix_idx = prefixes.index((1, 2))
        assert float(dedup_values[prefix_idx, 0]) == 12.0
        assert float(dedup_values[prefix_idx, 1]) == 13.0


class TestDedupProbsSum:
    """Test dedup_probs_sum function."""

    def test_sums_probabilities(self):
        """Test that probabilities for duplicate prefixes are summed."""
        inputs = jnp.array([[1, 2, 3], [1, 3, 2]])
        groups = make_prefix_groups(inputs)

        probs = jnp.array(
            [
                [0.2, 0.6, 0.2],  # sequence 0
                [0.2, 0.5, 0.3],  # sequence 1
            ]
        )
        dedup_probs, prefixes = dedup_probs_sum(probs, groups)
        prefix_idx = prefixes.index((1,))
        assert jnp.allclose(dedup_probs[prefix_idx], 0.2)

    def test_normalization(self):
        """Test that output probabilities sum to 1."""
        inputs = jnp.array([[1, 2], [3, 4]])
        groups = make_prefix_groups(inputs)

        probs = jnp.array(
            [
                [0.5, 0.5],  # sequence 0
                [0.2, 0.8],  # sequence 1
            ]
        )

        dedup_probs, _ = dedup_probs_sum(probs, groups)

        assert jnp.allclose(dedup_probs.sum(), 1.0)

    def test_preserves_order(self):
        """Test that the order of prefixes is preserved."""
        inputs = jnp.array([[1, 2], [1, 3], [1, 2]])
        groups = make_prefix_groups(inputs)

        probs = jnp.array(
            [
                [0.2, 0.8],  # sequence 0
                [0.5, 0.5],  # sequence 1
                [0.3, 0.7],  # sequence 2
            ]
        )

        _, prefixes = dedup_probs_sum(probs, groups)

        # Prefixes should be in the order they first appear
        expected_order = [(1,), (1, 2), (1, 3)]
        assert prefixes == expected_order


class TestDedupLastTokenTensorFirst:
    """Test dedup_last_token_tensor_first function."""

    def test_takes_first_sequence(self, simple_beliefs):
        """Test that first occurrence of duplicate sequence is selected."""
        inputs = jnp.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        groups = make_sequence_groups(inputs)

        # Get last token beliefs
        last_beliefs = simple_beliefs[:, -1, :]  # shape (3, 2)

        dedup_values, sequences = dedup_last_token_tensor_first(last_beliefs, groups)

        # Should have 2 unique sequences
        assert dedup_values.shape[0] == 2
        assert len(sequences) == 2

        # Sequence [1, 2, 3] should get value from first occurrence (index 0)
        seq_idx = sequences.index((1, 2, 3))
        assert jnp.allclose(dedup_values[seq_idx], simple_beliefs[0, -1, :])

    def test_preserves_dimensions(self):
        """Test that tensor dimensions are preserved."""
        inputs = jnp.array([[1, 2], [3, 4], [1, 2]])
        groups = make_sequence_groups(inputs)

        tensor = jnp.ones((3, 5))  # 3 sequences, 5 features

        dedup_values, _ = dedup_last_token_tensor_first(tensor, groups)

        # Should have 2 unique sequences, 5 features
        assert dedup_values.shape == (2, 5)


class TestDedupLastTokenProbsSum:
    """Test dedup_last_token_probs_sum function."""

    def test_sums_sequence_probabilities(self):
        """Test that probabilities for duplicate sequences are summed."""
        inputs = jnp.array([[1, 2], [3, 4], [1, 2]])
        groups = make_sequence_groups(inputs)

        probs = jnp.array([0.3, 0.4, 0.3])

        dedup_probs, sequences = dedup_last_token_probs_sum(probs, groups)

        # Sequence [1, 2] should sum: 0.3 + 0.3 = 0.6 (after normalization: 0.6/1.0)
        seq_idx = sequences.index((1, 2))
        assert jnp.allclose(dedup_probs[seq_idx], 0.6)

    def test_output_normalized(self):
        """Test that output probabilities sum to 1."""
        inputs = jnp.array([[1, 2], [3, 4], [5, 6]])
        groups = make_sequence_groups(inputs)

        probs = jnp.array([0.5, 0.3, 0.2])

        dedup_probs, _ = dedup_last_token_probs_sum(probs, groups)

        assert jnp.allclose(dedup_probs.sum(), 1.0)


class TestBuildPrefixDataset:
    """Test build_prefix_dataset integration."""

    def test_basic_functionality(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test basic prefix dataset building."""
        dataset = build_prefix_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Check that dataset has expected fields
        assert dataset.beliefs is not None
        assert dataset.probs is not None
        assert dataset.activations_by_layer is not None

        # Check probabilities sum to 1
        assert jnp.allclose(jnp.sum(dataset.probs), 1.0)

        # Check shapes are consistent

        assert isinstance(dataset.beliefs, jax.Array)
        n_prefixes = dataset.beliefs.shape[0]
        assert dataset.probs.shape[0] == n_prefixes
        for layer_acts in dataset.activations_by_layer.values():
            assert layer_acts.shape[0] == n_prefixes

    def test_preserves_layer_dimensions(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test that layer feature dimensions are preserved."""
        dataset = build_prefix_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Layer 0 should have 4 features
        assert dataset.activations_by_layer["layer_0"].shape[1] == 4

        # Layer 1 should have 6 features
        assert dataset.activations_by_layer["layer_1"].shape[1] == 6

    def test_belief_dimensions(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test that belief dimensions are preserved."""
        dataset = build_prefix_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Beliefs should have 2 dimensions (from fixture)
        assert isinstance(dataset.beliefs, jax.Array)
        assert dataset.beliefs.shape[1] == 2


class TestBuildLastTokenDataset:
    """Test build_last_token_dataset integration."""

    def test_basic_functionality(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test basic last token dataset building."""
        dataset = build_last_token_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Check that dataset has expected fields
        assert dataset.beliefs is not None
        assert dataset.probs is not None
        assert dataset.activations_by_layer is not None

        # Check probabilities sum to 1
        assert jnp.allclose(jnp.sum(dataset.probs), 1.0)

    def test_deduplication(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test that duplicate sequences are properly deduplicated."""
        # simple_inputs has sequences [1,2,3], [1,2,4], [1,2,3]
        # So we should get 2 unique sequences

        dataset = build_last_token_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Should have 2 unique sequences
        assert isinstance(dataset.beliefs, jax.Array)
        assert dataset.beliefs.shape[0] == 2
        assert dataset.probs.shape[0] == 2

    def test_preserves_dimensions(self, simple_inputs, simple_beliefs, simple_probs, simple_activations):
        """Test that feature dimensions are preserved."""
        dataset = build_last_token_dataset(simple_inputs, simple_beliefs, simple_probs, simple_activations)

        # Check belief dimension
        assert isinstance(dataset.beliefs, jax.Array)
        assert dataset.beliefs.shape[1] == 2

        # Check layer dimensions
        assert dataset.activations_by_layer["layer_0"].shape[1] == 4
        assert dataset.activations_by_layer["layer_1"].shape[1] == 6
