"""Tests for hidden factor functionality in FactoredGenerativeProcess."""

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    build_factored_process,
    build_factored_process_from_spec,
)
from simplexity.generative_processes.factored_generative_process import FactoredGenerativeProcess
from simplexity.generative_processes.structures import IndependentStructure, SequentialConditional


def _tensor_from_probs(variant_probs):
    """Create transition tensor from probability list."""
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


class TestHiddenFactorVocabSize:
    """Tests for vocab_size and full_vocab_size properties with hidden factors."""

    def test_single_hidden_factor(self):
        """A single hidden factor should reduce observable vocab size."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),  # V=2
            _tensor_from_probs([[0.7, 0.3]]),  # V=2
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        assert process.vocab_size == 2  # Only factor 0 observable
        assert process.full_vocab_size == 4  # Full joint 2 * 2

    def test_multiple_hidden_factors(self):
        """Multiple hidden factors should reduce observable vocab size accordingly."""
        component_types = ("hmm", "hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.5, 0.5]]),  # V=2
            _tensor_from_probs([[0.3, 0.4, 0.3]]),  # V=3
            _tensor_from_probs([[0.25, 0.25, 0.25, 0.25]]),  # V=4
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

        # Hide factors 0 and 2
        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({0, 2}),
        )

        assert process.vocab_size == 3  # Only factor 1 observable
        assert process.full_vocab_size == 2 * 3 * 4  # Full joint

    def test_no_hidden_factors_unchanged(self):
        """With no hidden factors, vocab_size should equal full_vocab_size."""
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

        # No hidden factors
        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
        )

        assert process.vocab_size == 4
        assert process.full_vocab_size == 4
        assert process.vocab_size == process.full_vocab_size


class TestHiddenFactorObservationDistribution:
    """Tests for observation_probability_distribution with hidden factors."""

    def test_marginalization_sums_to_one(self):
        """Marginalized distribution over observable tokens should sum to 1."""
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        dist = process.observation_probability_distribution(process.initial_state)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0)

    def test_observable_vocab_size_correct(self):
        """Distribution should have correct size for observable vocab."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.3, 0.4, 0.3]]),  # V=3
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),  # Hide factor 1
        )

        dist = process.observation_probability_distribution(process.initial_state)
        assert dist.shape == (2,)  # Only factor 0's vocab

    def test_marginalization_produces_correct_values(self):
        """Marginalizing should produce correct summed probabilities."""
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

        # Process without hidden factors
        process_full = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
        )

        # Process with factor 1 hidden
        process_hidden = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        full_dist = process_full.observation_probability_distribution(process_full.initial_state)
        hidden_dist = process_hidden.observation_probability_distribution(process_hidden.initial_state)

        # Full joint: [0.42, 0.18, 0.28, 0.12] for tokens (0,0), (0,1), (1,0), (1,1)
        # Marginalizing over factor 1: [0.42+0.18, 0.28+0.12] = [0.6, 0.4]
        expected = jnp.array([0.6, 0.4], dtype=jnp.float32)
        chex.assert_trees_all_close(hidden_dist, expected)


class TestHiddenFactorTransitions:
    """Tests for transition_states with hidden factors."""

    def test_hidden_factor_state_evolves(self):
        """Hidden factor states should evolve during transitions."""
        component_types = ("hmm", "hmm")
        transition_matrices = (
            jnp.array(
                [
                    [
                        [[0.9, 0.1], [0.2, 0.8]],
                        [[0.3, 0.7], [0.6, 0.4]],
                    ]
                ],
                dtype=jnp.float32,
            ),  # Observable
            jnp.array(
                [
                    [
                        [[0.7, 0.3], [0.4, 0.6]],
                        [[0.5, 0.5], [0.5, 0.5]],
                    ]
                ],
                dtype=jnp.float32,
            ),  # Hidden
        )
        normalizing_eigenvectors = (
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
        )
        initial_states = (
            jnp.array([0.5, 0.5], dtype=jnp.float32),
            jnp.array([0.5, 0.5], dtype=jnp.float32),
        )
        structure = IndependentStructure()

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        # Transition with observable token 0
        obs = jnp.array(0, dtype=jnp.int32)
        new_state = process.transition_states(process.initial_state, obs)

        # Both observable and hidden states should change
        assert not jnp.allclose(new_state[0], initial_states[0])
        assert not jnp.allclose(new_state[1], initial_states[1])

    def test_all_beliefs_returned(self):
        """transition_states should return beliefs for all factors including hidden."""
        component_types = ("hmm", "hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.7, 0.3]]),
            _tensor_from_probs([[0.5, 0.5]]),
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),  # Hide factor 1
        )

        # Observable vocab is 2 * 2 = 4 (factors 0 and 2)
        obs = jnp.array(0, dtype=jnp.int32)
        new_state = process.transition_states(process.initial_state, obs)

        # Should have 3 belief vectors (all factors)
        assert len(new_state) == 3
        assert len(process.initial_state) == 3


class TestHiddenFactorEmission:
    """Tests for emit_observation with hidden factors."""

    def test_emit_produces_observable_tokens(self):
        """emit_observation should produce tokens in observable vocab range."""
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        key = jax.random.PRNGKey(0)
        for i in range(10):
            token = process.emit_observation(process.initial_state, jax.random.fold_in(key, i))
            assert 0 <= int(token) < process.vocab_size


class TestHiddenFactorBuilder:
    """Tests for builder functions with hidden factors."""

    def test_spec_with_hidden_flag(self):
        """build_factored_process_from_spec should honor 'hidden' flag in spec."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
                "hidden": True,
            },
        ]

        process = build_factored_process_from_spec("independent", spec)

        assert process.hidden_factor_indices == frozenset({1})
        assert process.observable_factor_indices == (0,)
        assert process.vocab_size == 3  # Only factor 0's vocab (mess3 has vocab 3)
        assert process.full_vocab_size == 9  # 3 * 3

    def test_backwards_compatible_no_hidden(self):
        """Specs without 'hidden' flag should work as before."""
        spec = [
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
        ]

        process = build_factored_process_from_spec("independent", spec)

        assert process.hidden_factor_indices == frozenset()
        assert process.observable_factor_indices == (0, 1)
        assert process.vocab_size == 9
        assert process.full_vocab_size == 9

    def test_build_factored_process_with_hidden_indices(self):
        """build_factored_process should accept hidden_factor_indices parameter."""
        component_types = ["hmm", "hmm"]
        transition_matrices = [
            _tensor_from_probs([[0.6, 0.4]]),
            _tensor_from_probs([[0.7, 0.3]]),
        ]
        normalizing_eigenvectors = [
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ]
        initial_states = [
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        ]

        process = build_factored_process(
            structure_type="independent",
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            hidden_factor_indices=frozenset({0}),
        )

        assert process.hidden_factor_indices == frozenset({0})
        assert process.vocab_size == 2  # Only factor 1


class TestHiddenFactorWithChainStructure:
    """Tests for hidden factors with chain structure."""

    def test_hidden_factor_can_influence_visible_factor(self):
        """A hidden factor in a chain can influence visible factor variant selection."""
        # Factor 0 is hidden, factor 1 is observable and depends on factor 0
        component_types = ("hmm", "hmm")
        transition_matrices = (
            _tensor_from_probs([[0.8, 0.2]]),  # Hidden factor, V=2
            _tensor_from_probs([[0.9, 0.1], [0.1, 0.9]]),  # Observable, 2 variants, V=2
        )
        normalizing_eigenvectors = (
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
        )
        initial_states = (
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )

        # Control map: hidden factor token 0 -> variant 0, token 1 -> variant 1
        structure = SequentialConditional(
            control_maps=(None, jnp.array([0, 1], dtype=jnp.int32)),
            vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
        )

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({0}),
        )

        # Observable vocab is just factor 1's vocab (size 2)
        assert process.vocab_size == 2
        assert process.full_vocab_size == 4

        # Distribution should be a mixture depending on hidden factor
        dist = process.observation_probability_distribution(process.initial_state)
        assert dist.shape == (2,)
        chex.assert_trees_all_close(jnp.sum(dist), 1.0)

        # When hidden=0 (prob 0.8): obs dist is [0.9, 0.1]
        # When hidden=1 (prob 0.2): obs dist is [0.1, 0.9]
        # Marginal: 0.8 * [0.9, 0.1] + 0.2 * [0.1, 0.9] = [0.74, 0.26]
        expected = jnp.array([0.74, 0.26], dtype=jnp.float32)
        chex.assert_trees_all_close(dist, expected)


class TestHiddenFactorProbability:
    """Tests for probability computation with hidden factors."""

    def test_probability_with_hidden_factor(self):
        """probability() should work correctly with hidden factors."""
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        # Observable vocab is [0, 1] for factor 0
        observations = jnp.array([0, 1, 0], dtype=jnp.int32)
        prob = process.probability(observations)

        # Probability should be valid
        assert prob > 0
        assert prob <= 1

    def test_log_probability_with_hidden_factor(self):
        """log_probability() should work correctly with hidden factors."""
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

        process = FactoredGenerativeProcess(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            hidden_factor_indices=frozenset({1}),
        )

        observations = jnp.array([0, 1, 0], dtype=jnp.int32)
        log_prob = process.log_probability(observations)
        prob = process.probability(observations)

        chex.assert_trees_all_close(log_prob, jnp.log(prob))
