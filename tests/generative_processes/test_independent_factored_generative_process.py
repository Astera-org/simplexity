"""Tests for IndependentFactoredGenerativeProcess."""

import logging

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.independent_factored_generative_process import (
    IndependentFactoredGenerativeProcess,
)
from simplexity.generative_processes.structures import IndependentStructure, SequentialConditional


def _tensor_from_probs(variant_probs):
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


@pytest.fixture
def two_factor_independent_process():
    """Simple two-factor process with IndependentStructure."""
    component_types = ("hmm", "hmm")
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    structure = IndependentStructure()
    return IndependentFactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )


@pytest.fixture
def three_factor_process_with_frozen():
    """Three-factor process with factor 1 frozen."""
    component_types = ("hmm", "hmm", "hmm")
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3]]),
        _tensor_from_probs([[0.8, 0.2]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    structure = IndependentStructure()
    return IndependentFactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
        frozen_factor_indices=frozenset({1}),
        frozen_key=jax.random.PRNGKey(42),
    )


class TestEmitObservation:
    """Tests for emit_observation method."""

    def test_emit_observation_returns_valid_token(self, two_factor_independent_process):
        """emit_observation should return a valid composite token."""
        process = two_factor_independent_process
        token = process.emit_observation(process.initial_state, jax.random.PRNGKey(0))
        assert token.shape == ()
        assert 0 <= int(token) < process.vocab_size

    def test_emit_observation_samples_independently(self, two_factor_independent_process):
        """Samples should match product of marginal distributions."""
        process = two_factor_independent_process
        n_samples = 10000
        keys = jax.random.split(jax.random.PRNGKey(0), n_samples)

        tokens = jax.vmap(lambda k: process.emit_observation(process.initial_state, k))(keys)
        factor_tokens = process.encoder.extract_factors_vectorized(tokens)

        # Expected marginals: factor 0 is [0.6, 0.4], factor 1 is [0.7, 0.3]
        empirical_0 = jnp.bincount(factor_tokens[:, 0], length=2) / n_samples
        empirical_1 = jnp.bincount(factor_tokens[:, 1], length=2) / n_samples

        chex.assert_trees_all_close(empirical_0, jnp.array([0.6, 0.4]), atol=0.05)
        chex.assert_trees_all_close(empirical_1, jnp.array([0.7, 0.3]), atol=0.05)


class TestGenerate:
    """Tests for generate method."""

    def test_generate_produces_correct_shapes(self, two_factor_independent_process):
        """generate should produce correctly shaped outputs."""
        process = two_factor_independent_process
        batch_size = 4
        seq_len = 10

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        final_states, observations = process.generate(batch_states, keys, seq_len, False)

        assert observations.shape == (batch_size, seq_len)
        assert final_states[0].shape == (batch_size, 1)
        assert final_states[1].shape == (batch_size, 1)

    def test_generate_returns_all_states_when_requested(self, two_factor_independent_process):
        """generate with return_all_states=True should return state sequences."""
        process = two_factor_independent_process
        batch_size = 4
        seq_len = 10

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        all_states, observations = process.generate(batch_states, keys, seq_len, True)

        assert observations.shape == (batch_size, seq_len)
        assert all_states[0].shape == (batch_size, seq_len, 1)
        assert all_states[1].shape == (batch_size, seq_len, 1)


class TestFrozenFactors:
    """Tests for frozen factor behavior."""

    def test_frozen_factor_same_across_batch(self, three_factor_process_with_frozen):
        """Frozen factor should produce identical sequences across batch samples."""
        process = three_factor_process_with_frozen
        batch_size = 8
        seq_len = 20

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(123), batch_size)

        _, observations = process.generate(batch_states, keys, seq_len, False)
        factor_tokens = jax.vmap(process.encoder.extract_factors_vectorized)(observations)

        # Factor 1 (index 1) is frozen - should be identical across batch
        frozen_factor_sequences = factor_tokens[:, :, 1]
        for i in range(1, batch_size):
            chex.assert_trees_all_equal(frozen_factor_sequences[0], frozen_factor_sequences[i])

    def test_unfrozen_factors_vary_across_batch(self, three_factor_process_with_frozen):
        """Unfrozen factors should vary across batch samples."""
        process = three_factor_process_with_frozen
        batch_size = 8
        seq_len = 20

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(456), batch_size)

        _, observations = process.generate(batch_states, keys, seq_len, False)
        factor_tokens = jax.vmap(process.encoder.extract_factors_vectorized)(observations)

        # Factors 0 and 2 are unfrozen - should differ across batch
        unfrozen_0_sequences = factor_tokens[:, :, 0]
        unfrozen_2_sequences = factor_tokens[:, :, 2]

        # Check that not all samples are identical
        assert not jnp.all(unfrozen_0_sequences[0] == unfrozen_0_sequences[1])
        assert not jnp.all(unfrozen_2_sequences[0] == unfrozen_2_sequences[1])

    def test_frozen_sequences_reproducible(self, three_factor_process_with_frozen):
        """Frozen factor sequences should be reproducible across generate() calls."""
        process = three_factor_process_with_frozen
        batch_size = 4
        seq_len = 15

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)

        # First generation
        keys1 = jax.random.split(jax.random.PRNGKey(100), batch_size)
        _, obs1 = process.generate(batch_states, keys1, seq_len, False)
        factor_tokens1 = jax.vmap(process.encoder.extract_factors_vectorized)(obs1)

        # Second generation with different sample keys
        keys2 = jax.random.split(jax.random.PRNGKey(200), batch_size)
        _, obs2 = process.generate(batch_states, keys2, seq_len, False)
        factor_tokens2 = jax.vmap(process.encoder.extract_factors_vectorized)(obs2)

        # Frozen factor should be the same in both calls
        chex.assert_trees_all_equal(factor_tokens1[:, :, 1], factor_tokens2[:, :, 1])

    def test_all_factors_frozen(self):
        """With all factors frozen, all batch samples should be identical."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.7, 0.3]]),
        )
        normalizing_eigenvectors = (
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        )
        initial_states = (
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        structure = IndependentStructure()
        process = IndependentFactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            frozen_factor_indices=frozenset({0, 1}),
            frozen_key=jax.random.PRNGKey(999),
        )

        batch_size = 4
        seq_len = 10
        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        _, observations = process.generate(batch_states, keys, seq_len, False)

        # All batch samples should be identical
        for i in range(1, batch_size):
            chex.assert_trees_all_equal(observations[0], observations[i])

    def test_no_frozen_factors_matches_normal_behavior(self, two_factor_independent_process):
        """With no frozen factors, behavior should match normal generation."""
        process = two_factor_independent_process
        batch_size = 4
        seq_len = 10

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        _, observations = process.generate(batch_states, keys, seq_len, False)

        # All tokens should be valid
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < process.vocab_size)

        # Batch samples should differ (with high probability)
        assert not jnp.all(observations[0] == observations[1])


class TestValidation:
    """Tests for constructor validation."""

    def test_requires_frozen_key_when_frozen_indices_nonempty(self):
        """Should raise ValueError if frozen_factor_indices is non-empty but frozen_key is None."""
        component_types = ("hmm",)
        transition_matrices = (_tensor_from_probs([[0.6, 0.4]]),)
        normalizing_eigenvectors = (jnp.ones((1, 1), dtype=jnp.float32),)
        initial_states = (jnp.array([1.0], dtype=jnp.float32),)
        structure = IndependentStructure()

        with pytest.raises(ValueError, match="frozen_key is required"):
            IndependentFactoredGenerativeProcess(
                component_types=component_types,
                transition_matrices=transition_matrices,
                normalizing_eigenvectors=normalizing_eigenvectors,
                initial_states=initial_states,
                structure=structure,
                frozen_factor_indices=frozenset({0}),
                frozen_key=None,
            )

    def test_rejects_invalid_frozen_factor_index(self):
        """Should raise ValueError for out-of-range frozen factor indices."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.7, 0.3]]),
        )
        normalizing_eigenvectors = (
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        )
        initial_states = (
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        structure = IndependentStructure()

        with pytest.raises(ValueError, match="Invalid frozen factor index 5"):
            IndependentFactoredGenerativeProcess(
                component_types=component_types,
                transition_matrices=transition_matrices,
                normalizing_eigenvectors=normalizing_eigenvectors,
                initial_states=initial_states,
                structure=structure,
                frozen_factor_indices=frozenset({0, 5}),
                frozen_key=jax.random.PRNGKey(0),
            )

    def test_warns_for_non_independent_structure(self, caplog):
        """Should log warning when structure is not IndependentStructure."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
        )
        normalizing_eigenvectors = (
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
        )
        initial_states = (
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        structure = SequentialConditional(
            control_maps=(None, jnp.array([0, 1], dtype=jnp.int32)),
            vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
        )

        logger = logging.getLogger("simplexity")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.propagate = True
            IndependentFactoredGenerativeProcess(
                component_types=component_types,
                transition_matrices=transition_matrices,
                normalizing_eigenvectors=normalizing_eigenvectors,
                initial_states=initial_states,
                structure=structure,
            )

        assert "IndependentFactoredGenerativeProcess is designed for IndependentStructure" in caplog.text


class TestStateTransitions:  # pylint: disable=too-few-public-methods
    """Tests for state transitions with frozen factors."""

    def test_frozen_factor_states_match_across_batch(self, three_factor_process_with_frozen):
        """Frozen factor states should be identical across batch samples."""
        process = three_factor_process_with_frozen
        batch_size = 4
        seq_len = 10

        batch_states = tuple(jnp.tile(s[None, :], (batch_size, 1)) for s in process.initial_state)
        keys = jax.random.split(jax.random.PRNGKey(789), batch_size)

        all_states, _ = process.generate(batch_states, keys, seq_len, True)

        # Factor 1 states should be identical across batch
        frozen_factor_states = all_states[1]
        for i in range(1, batch_size):
            chex.assert_trees_all_close(frozen_factor_states[0], frozen_factor_states[i])
