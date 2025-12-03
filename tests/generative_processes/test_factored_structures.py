"""Tests for factored generative process conditional structures."""

import chex
import jax.numpy as jnp

from simplexity.generative_processes.factored_generative_process import ComponentType
from simplexity.generative_processes.structures import (
    ConditionalTransitions,
    FullyConditional,
    SequentialConditional,
)
from simplexity.generative_processes.structures.protocol import ConditionalContext


def _tensor_from_probs(variant_probs):
    """Convert per-variant emission probabilities into transition tensors."""
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


def _make_context(states, transition_matrices):
    """Helper to build a ConditionalContext for HMM components."""
    component_types: tuple[ComponentType, ...] = tuple("hmm" for _ in states)
    normalizing_eigenvectors = tuple(
        jnp.ones((tm.shape[0], tm.shape[-1]), dtype=jnp.float32) for tm in transition_matrices
    )
    vocab_sizes = jnp.array([tm.shape[1] for tm in transition_matrices])
    num_variants = tuple(int(tm.shape[0]) for tm in transition_matrices)
    return ConditionalContext(
        states=states,
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        vocab_sizes=vocab_sizes,
        num_variants=num_variants,
    )


def test_sequential_conditional_joint_distribution_and_variants():
    """SequentialConditional should respect the chain factorization."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    context = _make_context(states, transition_matrices)
    structure = SequentialConditional(control_maps=(None, jnp.array([0, 1], dtype=jnp.int32)))

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.42, 0.18, 0.08, 0.32], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)

    variants = structure.select_variants(
        (jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
        context,
    )
    chex.assert_trees_all_close(variants[0], jnp.array(0, dtype=jnp.int32))
    chex.assert_trees_all_close(variants[1], jnp.array(1, dtype=jnp.int32))


def test_fully_conditional_product_of_experts():
    """FullyConditional should build a normalized product-of-experts distribution."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.1, 0.9]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    context = _make_context(states, transition_matrices)
    structure = FullyConditional(
        control_maps=(jnp.array([0, 1], dtype=jnp.int32), jnp.array([1, 0], dtype=jnp.int32)),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.16, 0.10666667, 0.37333333, 0.36], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)

    variants = structure.select_variants(
        (jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32)),
        context,
    )
    chex.assert_trees_all_close(variants[0], jnp.array(1, dtype=jnp.int32))
    chex.assert_trees_all_close(variants[1], jnp.array(1, dtype=jnp.int32))


def test_conditional_transitions_with_independent_emissions():
    """ConditionalTransitions should reduce to independent emissions when no chain is given."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.2, 0.8]]),
        _tensor_from_probs([[0.5, 0.5], [0.1, 0.9]]),
    )
    context = _make_context(states, transition_matrices)
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([1, 0], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
        ),
        emission_variant_indices=jnp.array([1, 0], dtype=jnp.int32),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.1, 0.1, 0.4, 0.4], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)

    variants = structure.select_variants(
        (jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
        context,
    )
    chex.assert_trees_all_close(variants[0], jnp.array(1, dtype=jnp.int32))
    chex.assert_trees_all_close(variants[1], jnp.array(1, dtype=jnp.int32))


def test_conditional_transitions_with_sequential_emissions():
    """ConditionalTransitions should honor sequential emission control maps."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.2, 0.8]]),
        _tensor_from_probs([[0.9, 0.1], [0.3, 0.7]]),
    )
    context = _make_context(states, transition_matrices)
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([0, 0], dtype=jnp.int32),
            jnp.array([0, 0], dtype=jnp.int32),
        ),
        emission_variant_indices=jnp.array([0, 0], dtype=jnp.int32),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
        emission_control_maps=(None, jnp.array([1, 0], dtype=jnp.int32)),
    )

    assert structure.use_emission_chain is True

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.18, 0.42, 0.36, 0.04], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)
