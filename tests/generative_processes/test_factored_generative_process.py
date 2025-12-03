"""Integration tests for the factored generative process."""

import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.factored_generative_process import FactoredGenerativeProcess
from simplexity.generative_processes.structures import SequentialConditional
from simplexity.utils.factoring_utils import transition_with_obs


def _tensor_from_probs(variant_probs):
    arr = jnp.asarray(variant_probs, dtype=jnp.float32)
    return arr[..., None, None]


def _build_simple_chain_process():
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


def _build_multistate_chain_process():
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


def test_factored_process_observation_and_sequence_probability():
    """Observation, log-observation, and sequence probability APIs should agree."""
    process = _build_simple_chain_process()
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


def test_transition_states_uses_structure_selected_variants():
    """transition_states should respect variant selection from the structure."""
    process = _build_multistate_chain_process()
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


def test_emit_observation_respects_vocab_size():
    """Samples from emit_observation should be in the encoded vocabulary."""
    process = _build_simple_chain_process()
    token = process.emit_observation(process.initial_state, jax.random.PRNGKey(0))
    assert token.shape == ()
    assert 0 <= int(token) < process.vocab_size
