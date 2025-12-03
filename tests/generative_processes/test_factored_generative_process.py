"""Integration tests for the factored generative process."""

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.factored_generative_process import FactoredGenerativeProcess
from simplexity.generative_processes.structures import (
    ConditionalTransitions,
    FullyConditional,
    IndependentStructure,
    SequentialConditional,
)
from simplexity.utils.factoring_utils import transition_with_obs


def _tensor_from_probs(variant_probs):
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


@pytest.fixture
def simple_chain_process():
    """Simple two-factor chain with deterministic initial states."""
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
    return FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )


@pytest.fixture
def multistate_chain_process():
    """Two-factor chain with multi-state beliefs for transition testing."""
    component_types = ("hmm", "hmm")
    transition_matrices = (
        jnp.array(
            [
                [
                    [[0.9, 0.9], [0.1, 0.1]],
                    [[0.2, 0.2], [0.8, 0.8]],
                ]
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [[0.6, 0.6], [0.4, 0.4]],
                    [[0.3, 0.3], [0.7, 0.7]],
                ],
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.1, 0.1], [0.9, 0.9]],
                ],
            ],
            dtype=jnp.float32,
        ),
    )
    normalizing_eigenvectors = (
        jnp.ones((1, 2), dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([0.7, 0.3], dtype=jnp.float32),
        jnp.array([0.4, 0.6], dtype=jnp.float32),
    )
    structure = SequentialConditional(
        control_maps=(None, jnp.array([0, 1], dtype=jnp.int32)),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )
    return FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )


def _build_fully_conditional_process():
    component_types = ("hmm", "hmm")
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.1, 0.9]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    structure = FullyConditional(
        control_maps=(jnp.array([0, 1], dtype=jnp.int32), jnp.array([1, 0], dtype=jnp.int32)),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )
    return FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )


def _build_transition_coupled_process(*, emission_control_maps, emission_variant_indices):
    component_types = ("hmm", "hmm")
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.2, 0.8]]),
        _tensor_from_probs([[0.5, 0.5], [0.1, 0.9]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([0, 0], dtype=jnp.int32),
            jnp.array([1, 0], dtype=jnp.int32),
        ),
        emission_variant_indices=emission_variant_indices,
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
        emission_control_maps=emission_control_maps,
    )
    return FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )


def test_factored_process_observation_and_sequence_probability(simple_chain_process):
    """Observation, log-observation, and sequence probability APIs should agree."""
    process = simple_chain_process
    state = process.initial_state

    dist = process.observation_probability_distribution(state)
    expected = jnp.array([0.42, 0.18, 0.08, 0.32], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)
    chex.assert_trees_all_close(jnp.sum(dist), jnp.array(1.0, dtype=jnp.float32))

    log_state = tuple(jnp.log(s) for s in state)
    log_dist = process.log_observation_probability_distribution(log_state)
    chex.assert_trees_all_close(log_dist, jnp.log(dist))

    observations = jnp.array([0, 3, 1], dtype=jnp.int32)
    prob = process.probability(observations)
    expected_prob = jnp.array(0.42 * 0.32 * 0.18, dtype=jnp.float32)
    chex.assert_trees_all_close(prob, expected_prob)

    log_prob = process.log_probability(observations)
    chex.assert_trees_all_close(log_prob, jnp.log(expected_prob))


def test_transition_states_uses_structure_selected_variants(multistate_chain_process):
    """transition_states should respect variant selection from the structure."""
    process = multistate_chain_process
    state = process.initial_state
    obs_token = jnp.array(2, dtype=jnp.int32)  # (factor0=1, factor1=0)

    new_state = process.transition_states(state, obs_token)
    obs_tuple = process.encoder.token_to_tuple(obs_token)

    expected0 = transition_with_obs(
        process.component_types[0],
        state[0],
        process.transition_matrices[0][0],
        obs_tuple[0],
        None,
    )
    context = process._make_context(state)
    variant1 = int(process.structure.select_variants(obs_tuple, context)[1])
    expected1 = transition_with_obs(
        process.component_types[1],
        state[1],
        process.transition_matrices[1][variant1],
        obs_tuple[1],
        None,
    )

    chex.assert_trees_all_close(new_state[0], expected0)
    chex.assert_trees_all_close(new_state[1], expected1)


def test_emit_observation_respects_vocab_size(simple_chain_process):
    """Samples from emit_observation should be in the encoded vocabulary."""
    token = simple_chain_process.emit_observation(simple_chain_process.initial_state, jax.random.PRNGKey(0))
    assert token.shape == ()
    assert 0 <= int(token) < simple_chain_process.vocab_size


def test_fully_conditional_process_observation_distribution():
    """Fully conditional topology should yield expected product-of-experts distribution."""
    process = _build_fully_conditional_process()
    dist = process.observation_probability_distribution(process.initial_state)
    expected = jnp.array([0.16, 0.10666667, 0.37333333, 0.36], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_fully_conditional_transition_variants_follow_control_maps():
    """transition_states should honor fully-conditional variant selection."""
    component_types = ("hmm", "hmm")
    transition_matrices = (
        jnp.array(
            [
                [
                    [[0.8, 0.2], [0.0, 0.0]],
                    [[0.2, 0.8], [0.0, 0.0]],
                ],
                [
                    [[0.1, 0.9], [0.0, 0.0]],
                    [[0.9, 0.1], [0.0, 0.0]],
                ],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [[0.3, 0.7], [0.0, 0.0]],
                    [[0.6, 0.4], [0.0, 0.0]],
                ],
                [
                    [[0.4, 0.6], [0.0, 0.0]],
                    [[0.7, 0.3], [0.0, 0.0]],
                ],
            ],
            dtype=jnp.float32,
        ),
    )
    normalizing_eigenvectors = (
        jnp.ones((2, 2), dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([0.6, 0.4], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    )
    structure = FullyConditional(
        control_maps=(jnp.array([0, 1], dtype=jnp.int32), jnp.array([1, 0], dtype=jnp.int32)),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )
    process = FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )
    obs_tuple = (jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
    obs_token = process.encoder.tuple_to_token(obs_tuple)
    context = process._make_context(initial_states)
    variants = process.structure.select_variants(obs_tuple, context)
    new_state = process.transition_states(initial_states, obs_token)
    expected = (
        transition_with_obs("hmm", initial_states[0], transition_matrices[0][int(variants[0])], obs_tuple[0], None),
        transition_with_obs("hmm", initial_states[1], transition_matrices[1][int(variants[1])], obs_tuple[1], None),
    )
    chex.assert_trees_all_close(new_state[0], expected[0])
    chex.assert_trees_all_close(new_state[1], expected[1])


def test_transition_coupled_independent_emissions_distribution():
    """ConditionalTransitions with fixed emission variants should factorize emissions."""
    process = _build_transition_coupled_process(
        emission_control_maps=None,
        emission_variant_indices=jnp.array([1, 0], dtype=jnp.int32),
    )
    dist = process.observation_probability_distribution(process.initial_state)
    expected = jnp.array([0.1, 0.1, 0.4, 0.4], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_transition_coupled_sequential_emissions_distribution():
    """ConditionalTransitions with emission control maps should follow sequential chain."""
    process = _build_transition_coupled_process(
        emission_control_maps=(None, jnp.array([1, 0], dtype=jnp.int32)),
        emission_variant_indices=jnp.array([0, 0], dtype=jnp.int32),
    )
    dist = process.observation_probability_distribution(process.initial_state)
    expected = jnp.array([0.06, 0.54, 0.20, 0.20], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_transition_coupled_transition_variants_follow_control_maps():
    """Transition variants should depend on other-factor tokens as configured."""
    component_types = ("hmm", "hmm")
    transition_matrices = (
        jnp.array(
            [
                [
                    [[0.7, 0.3], [0.2, 0.8]],
                    [[0.5, 0.5], [0.4, 0.6]],
                ],
                [
                    [[0.4, 0.6], [0.3, 0.7]],
                    [[0.2, 0.8], [0.6, 0.4]],
                ],
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                [
                    [[0.6, 0.4], [0.1, 0.9]],
                    [[0.3, 0.7], [0.8, 0.2]],
                ],
                [
                    [[0.9, 0.1], [0.2, 0.8]],
                    [[0.5, 0.5], [0.7, 0.3]],
                ],
            ],
            dtype=jnp.float32,
        ),
    )
    normalizing_eigenvectors = (
        jnp.ones((2, 2), dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    initial_states = (
        jnp.array([0.8, 0.2], dtype=jnp.float32),
        jnp.array([0.3, 0.7], dtype=jnp.float32),
    )
    structure = ConditionalTransitions(
        control_maps_transition=(
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([1, 0], dtype=jnp.int32),
        ),
        emission_variant_indices=jnp.array([0, 1], dtype=jnp.int32),
        vocab_sizes=jnp.array([2, 2], dtype=jnp.int32),
    )
    process = FactoredGenerativeProcess(
        component_types=component_types,
        transition_matrices=transition_matrices,
        normalizing_eigenvectors=normalizing_eigenvectors,
        initial_states=initial_states,
        structure=structure,
    )
    obs_tuple = (jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
    obs_token = process.encoder.tuple_to_token(obs_tuple)
    context = process._make_context(initial_states)
    variants = process.structure.select_variants(obs_tuple, context)
    new_state = process.transition_states(initial_states, obs_token)
    expected = (
        transition_with_obs("hmm", initial_states[0], transition_matrices[0][int(variants[0])], obs_tuple[0], None),
        transition_with_obs("hmm", initial_states[1], transition_matrices[1][int(variants[1])], obs_tuple[1], None),
    )
    chex.assert_trees_all_close(new_state[0], expected[0])
    chex.assert_trees_all_close(new_state[1], expected[1])


def test_independent_structure_observation_distribution():
    """IndependentStructure should produce product of independent factor distributions."""
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
    )
    dist = process.observation_probability_distribution(process.initial_state)
    expected = jnp.array([0.42, 0.18, 0.28, 0.12], dtype=jnp.float32)
    chex.assert_trees_all_close(dist, expected)


def test_independent_structure_always_uses_variant_zero():
    """IndependentStructure should always select variant 0 for all factors."""
    component_types = ("hmm", "hmm")
    transition_matrices = (
        _tensor_from_probs([[0.6, 0.4], [0.1, 0.9]]),
        _tensor_from_probs([[0.7, 0.3], [0.2, 0.8]]),
    )
    normalizing_eigenvectors = (
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
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
    )
    obs_tuple = (jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
    obs_token = process.encoder.tuple_to_token(obs_tuple)
    context = process._make_context(initial_states)
    variants = process.structure.select_variants(obs_tuple, context)
    chex.assert_trees_all_equal(variants, (jnp.array(0), jnp.array(0)))
    new_state = process.transition_states(initial_states, obs_token)
    expected = (
        transition_with_obs("hmm", initial_states[0], transition_matrices[0][0], obs_tuple[0], None),
        transition_with_obs("hmm", initial_states[1], transition_matrices[1][0], obs_tuple[1], None),
    )
    chex.assert_trees_all_close(new_state[0], expected[0])
    chex.assert_trees_all_close(new_state[1], expected[1])


def test_independent_structure_get_required_params():
    """IndependentStructure should have no required params."""
    structure = IndependentStructure()
    required_params = structure.get_required_params()
    assert required_params == {}
