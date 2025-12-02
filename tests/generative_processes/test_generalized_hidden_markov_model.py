"""Tests for generalized hidden Markov models."""

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

from simplexity.generative_processes.builder import build_generalized_hidden_markov_model
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from tests.array_with_patchable_device import (
    ArrayWithPatchableDevice,
    patch_jax_for_patchable_device,
)
from tests.assertions import assert_proportional


@pytest.fixture
def z1r() -> GeneralizedHiddenMarkovModel:
    """Return the zero-one random generalized HMM."""
    return build_generalized_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})


@pytest.fixture
def fanizza_model() -> GeneralizedHiddenMarkovModel:
    """Return the fanizza generalized HMM."""
    return build_generalized_hidden_markov_model(process_name="fanizza", process_params={"alpha": 2000, "lamb": 0.49})


@pytest.mark.parametrize(("model_name", "vocab_size", "num_states"), [("z1r", 2, 3), ("fanizza_model", 2, 4)])
def test_properties(model_name: str, vocab_size: int, num_states: int, request: pytest.FixtureRequest):
    """Test vocab size and state count for the available fixtures."""
    model: GeneralizedHiddenMarkovModel = request.getfixturevalue(model_name)
    assert model.vocab_size == vocab_size
    assert model.num_states == num_states


def test_init_device_mismatch():
    """Test that transition matrices are moved to the model device if they are on a different device."""
    mock_cpu = create_autospec(jax.Device, instance=True)
    mock_gpu = create_autospec(jax.Device, instance=True)

    # Create real arrays for the actual computation
    # For GHMM, transition matrices must be 3D: (vocab_size, num_states, num_states)
    real_transition_matrices = jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    real_initial_state = jnp.array([0.5, 0.5])

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
        patch(
            "simplexity.generative_processes.generalized_hidden_markov_model.resolve_jax_device",
            return_value=mock_gpu,
        ),
        patch(
            "simplexity.generative_processes.generalized_hidden_markov_model.jax.device_put",
            side_effect=device_put_side_effect,
        ),
        patch_jax_for_patchable_device(
            "simplexity.generative_processes.generalized_hidden_markov_model",
            mock_devices=(mock_cpu, mock_gpu),
        ),
        patch(
            "simplexity.generative_processes.generalized_hidden_markov_model.SIMPLEXITY_LOGGER.warning"
        ) as mock_warning,
    ):
        assert transition_matrices_on_cpu.device == mock_cpu
        assert initial_state_on_cpu.device == mock_cpu
        model = GeneralizedHiddenMarkovModel(transition_matrices_on_cpu, initial_state_on_cpu, device="gpu")
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


def test_normalize_belief_state(z1r: GeneralizedHiddenMarkovModel):
    """Test belief-state normalization under linear probabilities."""
    state = jnp.array([2, 5, 1])
    belief_state = z1r.normalize_belief_state(state)
    chex.assert_trees_all_close(belief_state, jnp.array([0.25, 0.625, 0.125]))

    state = jnp.array([0, 0, 0])
    belief_state = z1r.normalize_belief_state(state)
    assert jnp.all(jnp.isnan(belief_state))


def test_normalize_log_belief_state(z1r: GeneralizedHiddenMarkovModel):
    """Test log-space belief-state normalization."""
    state = jnp.log(jnp.array([2, 5, 1]))
    log_belief_state = z1r.normalize_log_belief_state(state)
    chex.assert_trees_all_close(log_belief_state, jnp.log(jnp.array([0.25, 0.625, 0.125])))

    log_belief_state = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
    log_belief_state = z1r.normalize_log_belief_state(log_belief_state)
    assert jnp.all(jnp.isnan(log_belief_state))


def test_hmm_single_transition(z1r: GeneralizedHiddenMarkovModel):
    """Test single-step transitions from each canonical state."""
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


@pytest.mark.parametrize("model_name", ["z1r", "fanizza_model"])
def test_generate(model_name: str, request: pytest.FixtureRequest):
    """Test sampling sequences without intermediate states."""
    model: GeneralizedHiddenMarkovModel = request.getfixturevalue(model_name)
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(model.initial_state[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = model.generate(initial_states, keys, sequence_len, False)
    assert intermediate_states.shape == (batch_size, model.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = model.generate(intermediate_states, keys, sequence_len, False)
    assert final_states.shape == (batch_size, model.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


@pytest.mark.parametrize("model_name", ["z1r", "fanizza_model"])
def test_generate_with_intermediate_states(model_name: str, request: pytest.FixtureRequest):
    """Test sampling sequences while also returning intermediates."""
    model: GeneralizedHiddenMarkovModel = request.getfixturevalue(model_name)
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(model.initial_state[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, observations = model.generate(initial_states, keys, sequence_len, True)
    assert intermediate_states.shape == (batch_size, sequence_len, model.num_states)
    assert observations.shape == (batch_size, sequence_len)
    last_intermediate_states = intermediate_states[:, -1, :]

    final_states, observations = model.generate(last_intermediate_states, keys, sequence_len, True)
    assert final_states.shape == (batch_size, sequence_len, model.num_states)
    assert observations.shape == (batch_size, sequence_len)


def test_hmm_observation_probability_distribution(z1r: GeneralizedHiddenMarkovModel):
    """Test observation probability distribution in probability space."""
    state = jnp.array([0.3, 0.1, 0.6])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))

    state = jnp.array([0.5, 0.3, 0.2])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))


def test_ghmm_observation_probability_distribution(fanizza_model: GeneralizedHiddenMarkovModel):
    """Test observation probabilities for the fanizza GHMM."""
    valid_state = fanizza_model.initial_state
    obs_probs = fanizza_model.observation_probability_distribution(valid_state)
    assert jnp.isclose(jnp.sum(obs_probs), 1)
    assert jnp.all(obs_probs >= 0)
    assert jnp.all(obs_probs <= 1)

    invalid_state = jnp.array([0.3, 0.1, 0.6, 0.0])
    obs_probs = fanizza_model.observation_probability_distribution(invalid_state)
    assert jnp.isclose(jnp.sum(obs_probs), 1)
    assert jnp.logical_or(jnp.any(obs_probs < 0), jnp.any(obs_probs > 1))


def test_hmm_log_observation_probability_distribution(z1r: GeneralizedHiddenMarkovModel):
    """Test observation probabilities in log space."""
    log_belief_state = jnp.log(jnp.array([0.3, 0.1, 0.6]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=2e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))

    log_belief_state = jnp.log(jnp.array([0.5, 0.3, 0.2]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=2e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))


def test_hmm_probability(z1r: GeneralizedHiddenMarkovModel):
    """Test forward probability of a fixed observation sequence."""
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)


def test_ghmm_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    """Test probability bounds for random observations."""
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.vocab_size)

    probability = fanizza_model.probability(observations)
    assert 0 <= probability <= 1


def test_hmm_log_probability(z1r: GeneralizedHiddenMarkovModel):
    """Test log-probability of a fixed observation sequence."""
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))


def test_ghmm_log_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    """Test that log probabilities remain non-positive."""
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.vocab_size)

    log_probability = fanizza_model.log_probability(observations)
    assert log_probability <= 0
