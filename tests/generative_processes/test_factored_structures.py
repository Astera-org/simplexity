"""Tests for factored generative process conditional structures."""

import chex
import jax.numpy as jnp

from simplexity.generative_processes.structures import (
    ConditionalTransitions,
    FullyConditional,
    IndependentStructure,
    SequentialConditional,
)
from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import ComponentType


def _tensor_from_probs(variant_probs):
    """Convert per-variant emission probabilities into transition tensors."""
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


def _make_context(states, transition_matrices):
    """Helper building a ConditionalContext for HMM components."""
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


def test_independent_structure_joint_distribution():
    """IndependentStructure should compute product of independent marginals."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3]]),
    )
    context = _make_context(states, transition_matrices)
    structure = IndependentStructure()

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.42, 0.18, 0.28, 0.12], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_independent_structure_select_variants_always_zero():
    """IndependentStructure should always select variant 0 for all factors."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.1, 0.9]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    context = _make_context(states, transition_matrices)
    structure = IndependentStructure()

    variants = structure.select_variants(
        (jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
        context,
    )
    chex.assert_trees_all_close(variants[0], jnp.array(0, dtype=jnp.int32))
    chex.assert_trees_all_close(variants[1], jnp.array(0, dtype=jnp.int32))


def test_independent_structure_get_required_params():
    """IndependentStructure should have no required params."""
    structure = IndependentStructure()
    required_params = structure.get_required_params()
    assert required_params == {}


def test_independent_structure_with_three_factors():
    """IndependentStructure should handle three or more factors."""
    states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    transition_matrices = (
        _tensor_from_probs([[0.5, 0.5]]),
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3]]),
    )
    context = _make_context(states, transition_matrices)
    structure = IndependentStructure()

    dist = structure.compute_joint_distribution(context)
    # Joint = P(t0) * P(t1) * P(t2)
    # For (t0, t1, t2): 0.5 * 0.6 * 0.7 = 0.21 (for 000), etc.
    expected = jnp.array([0.21, 0.09, 0.14, 0.06, 0.21, 0.09, 0.14, 0.06], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected, atol=1e-6)


def test_sequential_conditional_with_three_factors():
    """SequentialConditional should handle chains with three or more factors."""
    states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.8, 0.2], [0.3, 0.7]]),
        _tensor_from_probs([[0.9, 0.1], [0.5, 0.5]]),
    )
    context = _make_context(states, transition_matrices)
    structure = SequentialConditional(
        control_maps=(
            None,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([1, 0], dtype=jnp.int32),
        ),
        vocab_sizes=jnp.array([2, 2, 2]),
    )

    dist = structure.compute_joint_distribution(context)
    assert dist.shape == (8,)
    chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)


def test_sequential_conditional_without_vocab_sizes():
    """SequentialConditional should work without precomputed vocab_sizes_py."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    context = _make_context(states, transition_matrices)
    # Don't pass vocab_sizes to constructor
    structure = SequentialConditional(control_maps=(None, jnp.array([0, 1], dtype=jnp.int32)))

    dist = structure.compute_joint_distribution(context)
    expected = jnp.array([0.42, 0.18, 0.08, 0.32], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_sequential_conditional_get_required_params():
    """SequentialConditional should return required params."""
    structure = SequentialConditional(control_maps=(None,))
    required_params = structure.get_required_params()
    assert required_params == {"control_maps": tuple}


def test_fully_conditional_get_required_params():
    """FullyConditional should return required params."""
    structure = FullyConditional(
        control_maps=(jnp.array([0], dtype=jnp.int32),), vocab_sizes=jnp.array([2])
    )
    required_params = structure.get_required_params()
    assert required_params == {"control_maps": tuple, "vocab_sizes": jnp.ndarray}


def test_fully_conditional_with_zero_normalization():
    """FullyConditional should handle zero normalization edge case."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    # Create distributions that will multiply to zero
    transition_matrices = (
        _tensor_from_probs([[0.0, 0.0], [1.0, 0.0]]),
        _tensor_from_probs([[1.0, 0.0], [0.0, 1.0]]),
    )
    context = _make_context(states, transition_matrices)
    structure = FullyConditional(
        control_maps=(jnp.array([0, 1], dtype=jnp.int32), jnp.array([1, 0], dtype=jnp.int32)),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )

    dist = structure.compute_joint_distribution(context)
    # Should fall back to uniform distribution when Z=0
    assert dist.shape == (4,)
    assert jnp.all(dist >= 0.0)


def test_conditional_transitions_get_required_params():
    """ConditionalTransitions should return required params."""
    structure = ConditionalTransitions(
        control_maps_transition=(jnp.array([0], dtype=jnp.int32),),
        emission_variant_indices=jnp.array([0]),
        vocab_sizes=jnp.array([2]),
    )
    required_params = structure.get_required_params()
    assert "control_maps_transition" in required_params
    assert "emission_variant_indices" in required_params
    assert "vocab_sizes" in required_params


def test_conditional_transitions_with_none_emission_maps():
    """ConditionalTransitions should handle None emission_control_maps."""
    states = (jnp.array([1.0], dtype=jnp.float32), jnp.array([1.0], dtype=jnp.float32))
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.2, 0.8]]),
        _tensor_from_probs([[0.5, 0.5], [0.1, 0.9]]),
    )
    context = _make_context(states, transition_matrices)
    # Don't pass emission_control_maps at all
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([1, 0], dtype=jnp.int32),
            jnp.array([0, 1], dtype=jnp.int32),
        ),
        emission_variant_indices=jnp.array([0, 1], dtype=jnp.int32),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )

    assert structure.use_emission_chain is False
    dist = structure.compute_joint_distribution(context)
    assert dist.shape == (4,)
    chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)


def test_conditional_transitions_sequential_with_none_in_chain():
    """ConditionalTransitions should handle None entries in sequential emission maps."""
    states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4]]),
        _tensor_from_probs([[0.8, 0.2], [0.3, 0.7]]),
        _tensor_from_probs([[0.9, 0.1], [0.5, 0.5], [0.7, 0.3], [0.4, 0.6]]),
    )
    context = _make_context(states, transition_matrices)
    # Sequential emissions: factor 0 fixed, factor 1 uses map, factor 2 uses different map
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([0, 0, 0], dtype=jnp.int32),
            jnp.array([0, 0, 0], dtype=jnp.int32),
            jnp.array([0, 0, 0], dtype=jnp.int32),
        ),
        emission_variant_indices=jnp.array([0, 0, 0], dtype=jnp.int32),
        vocab_sizes=jnp.array([2, 2, 2], dtype=jnp.int32),
        emission_control_maps=(
            None,
            None,
            jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        ),
    )

    assert structure.use_emission_chain is True
    dist = structure.compute_joint_distribution(context)
    assert dist.shape == (8,)
    chex.assert_trees_all_close(jnp.sum(dist), 1.0, atol=1e-6)
