"""Tests for standard hidden Markov models."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from typing import cast
from unittest.mock import call, create_autospec, patch

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from tests.array_with_patchable_device import (
    ArrayWithPatchableDevice,
    patch_jax_for_patchable_device,
)
from tests.assertions import assert_proportional


@pytest.fixture
def z1r() -> HiddenMarkovModel:
    """Return the zero-one random HMM."""
    return build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})


def test_properties(z1r: HiddenMarkovModel):
    """Test key properties of the base HMM."""
    assert z1r.vocab_size == 2
    assert z1r.num_states == 3
    assert_proportional(z1r.normalizing_eigenvector, jnp.ones(3))
    assert_proportional(z1r.initial_state, jnp.ones(3))


def test_init_device_mismatch():
    """Test that transition matrices are moved to the model device if they are on a different device."""
    mock_cpu = create_autospec(jax.Device, instance=True)
    mock_gpu = create_autospec(jax.Device, instance=True)

    # Create real arrays for the actual computation
    real_transition_matrices = jnp.array([[[0.5]], [[0.5]]])
    real_initial_state = jnp.array([1.0])

    # Create arrays that appear to be on CPU initially
    transition_matrices_on_cpu = cast(jax.Array, ArrayWithPatchableDevice(real_transition_matrices, mock_cpu))
    initial_state_on_cpu = cast(jax.Array, ArrayWithPatchableDevice(real_initial_state, mock_cpu))

    # Create arrays that appear to be on GPU after device_put
    transition_matrices_on_gpu = cast(jax.Array, ArrayWithPatchableDevice(real_transition_matrices, mock_gpu))
    initial_state_on_gpu = cast(jax.Array, ArrayWithPatchableDevice(real_initial_state, mock_gpu))

    def device_put_side_effect(array, device):
        """Mock device_put to return arrays with GPU device."""
        if array is transition_matrices_on_cpu:
            return transition_matrices_on_gpu
        if array is initial_state_on_cpu:
            return initial_state_on_gpu
        return array

    with (
        patch("simplexity.generative_processes.hidden_markov_model.resolve_jax_device", return_value=mock_gpu),
        patch(
            "simplexity.generative_processes.hidden_markov_model.jax.device_put",
            side_effect=device_put_side_effect,
        ),
        patch_jax_for_patchable_device(
            "simplexity.generative_processes.hidden_markov_model", mock_devices=(mock_cpu, mock_gpu)
        ),
        patch("simplexity.generative_processes.hidden_markov_model.SIMPLEXITY_LOGGER.warning") as mock_warning,
    ):
        assert transition_matrices_on_cpu.device == mock_cpu
        assert initial_state_on_cpu.device == mock_cpu
        model = HiddenMarkovModel(transition_matrices_on_cpu, initial_state_on_cpu, device="gpu")
        assert model.transition_matrices.device == mock_gpu
        assert model.initial_state.device == mock_gpu
        mock_warning.assert_has_calls(
            [
                call(
                    "Transition matrices are on device %s but model is on device %s. "
                    "Moving transition matrices to model device.",
                    mock_cpu,
                    mock_gpu,
                ),
                call(
                    "Initial state is on device %s but model is on device %s. Moving initial state to model device.",
                    mock_cpu,
                    mock_gpu,
                ),
            ]
        )


def test_normalize_belief_state(z1r: HiddenMarkovModel):
    """Test normalization in probability space."""
    state = jnp.array([2, 5, 1])
    belief_state = z1r.normalize_belief_state(state)
    chex.assert_trees_all_close(belief_state, jnp.array([0.25, 0.625, 0.125]))

    state = jnp.array([0, 0, 0])
    belief_state = z1r.normalize_belief_state(state)
    assert jnp.all(jnp.isnan(belief_state))


def test_normalize_log_belief_state(z1r: HiddenMarkovModel):
    """Test normalization in log space."""
    state = jnp.log(jnp.array([2, 5, 1]))
    log_belief_state = z1r.normalize_log_belief_state(state)
    chex.assert_trees_all_close(log_belief_state, jnp.log(jnp.array([0.25, 0.625, 0.125])))

    log_belief_state = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
    log_belief_state = z1r.normalize_log_belief_state(log_belief_state)
    assert jnp.all(jnp.isnan(log_belief_state))


def test_single_transition(z1r: HiddenMarkovModel):
    """Test single transition outcomes and observations."""
    zero_state = jnp.array([[1.0, 0.0, 0.0]])
    one_state = jnp.array([[0.0, 1.0, 0.0]])
    random_state = jnp.array([[0.0, 0.0, 1.0]])

    probability = eqx.filter_vmap(z1r.normalize_belief_state)

    key = jax.random.PRNGKey(0)[None, :]
    single_transition = 1

    next_state, observation = z1r.generate(zero_state, key, single_transition, False)
    assert_proportional(probability(next_state), one_state)
    assert observation == jnp.array(0)

    next_state, observation = z1r.generate(one_state, key, single_transition, False)
    assert_proportional(probability(next_state), random_state)
    assert observation == jnp.array(1)

    next_state, observation = z1r.generate(random_state, key, single_transition, False)
    assert_proportional(probability(next_state), zero_state)

    mixed_state = jnp.array([[0.4, 0.4, 0.2]])

    next_state, observation = z1r.generate(mixed_state, key, single_transition, False)
    # P(next=0 | obs=x) = P(prev=2 | obs=x)
    # P(next=1 | obs=x) = P(prev=0 | obs=x)
    # P(next=2 | obs=x) = P(prev=1 | obs=x)
    if observation == 0:
        # P(obs=0 | prev=2) * P(prev=2) = 0.5 * 0.2 = 0.1
        # P(obs=0 | prev=0) * P(prev=0) = 1.0 * 0.4 = 0.4
        # P(obs=0 | prev=1) * P(prev=1) = 0.0 * 0.4 = 0.0
        next_mixed_state = jnp.array([[0.2, 0.8, 0.0]])
    else:
        # P(obs=1 | prev=2) * P(prev=2) = 0.5 * 0.2 = 0.1
        # P(obs=1 | prev=0) * P(prev=0) = 0.0 * 0.4 = 0.0
        # P(obs=1 | prev=1) * P(prev=1) = 1.0 * 0.4 = 0.4
        next_mixed_state = jnp.array([[0.2, 0.0, 0.8]])
    assert_proportional(probability(next_state), next_mixed_state)


def test_generate(z1r: HiddenMarkovModel):
    """Test multi-step generation without intermediates."""
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(z1r.normalizing_eigenvector[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = z1r.generate(initial_states, keys, sequence_len, False)
    assert intermediate_states.shape == (batch_size, z1r.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = z1r.generate(intermediate_states, keys, sequence_len, False)
    assert final_states.shape == (batch_size, z1r.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


def test_observation_probability_distribution(z1r: HiddenMarkovModel):
    """Test probability-space observation distribution."""
    state = jnp.array([0.3, 0.1, 0.6])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))

    state = jnp.array([0.5, 0.3, 0.2])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))


def test_log_observation_probability_distribution(z1r: HiddenMarkovModel):
    """Test log-space observation distribution."""
    log_belief_state = jnp.log(jnp.array([0.3, 0.1, 0.6]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=1e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))

    log_belief_state = jnp.log(jnp.array([0.5, 0.3, 0.2]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=1e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))


def test_probability(z1r: HiddenMarkovModel):
    """Test probability of a fixed observation sequence."""
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)


def test_log_probability(z1r: HiddenMarkovModel):
    """Test log probability of a fixed observation sequence."""
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))
