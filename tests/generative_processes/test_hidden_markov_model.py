import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from tests.assertions import assert_proportional


@pytest.fixture
def z1r() -> HiddenMarkovModel:
    return build_hidden_markov_model("zero_one_random", p=0.5)


def test_properties(z1r: HiddenMarkovModel):
    assert z1r.vocab_size == 2
    assert z1r.num_states == 3
    assert_proportional(z1r.normalizing_eigenvector, jnp.ones(3))
    assert_proportional(z1r.state_eigenvector, jnp.ones(3))


def test_normalize_belief_state(z1r: HiddenMarkovModel):
    state = jnp.array([2, 5, 1])
    belief_state = z1r.normalize_belief_state(state)
    chex.assert_trees_all_close(belief_state, jnp.array([0.25, 0.625, 0.125]))

    state = jnp.array([0, 0, 0])
    belief_state = z1r.normalize_belief_state(state)
    assert jnp.all(jnp.isnan(belief_state))


def test_normalize_log_belief_state(z1r: HiddenMarkovModel):
    state = jnp.log(jnp.array([2, 5, 1]))
    log_belief_state = z1r.normalize_log_belief_state(state)
    chex.assert_trees_all_close(log_belief_state, jnp.log(jnp.array([0.25, 0.625, 0.125])))

    log_state = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
    log_belief_state = z1r.normalize_log_belief_state(log_state)
    assert jnp.all(jnp.isnan(log_belief_state))


def test_single_transition(z1r: HiddenMarkovModel):
    zero_state = jnp.array([[1.0, 0.0, 0.0]])
    one_state = jnp.array([[0.0, 1.0, 0.0]])
    random_state = jnp.array([[0.0, 0.0, 1.0]])

    probability = eqx.filter_vmap(z1r.normalize_belief_state)

    key = jax.random.PRNGKey(0)[None, :]
    single_transition = 1

    next_state, observation = z1r.generate(zero_state, key, single_transition)
    assert_proportional(probability(next_state), one_state)
    assert observation == jnp.array(0)

    next_state, observation = z1r.generate(one_state, key, single_transition)
    assert_proportional(probability(next_state), random_state)
    assert observation == jnp.array(1)

    next_state, observation = z1r.generate(random_state, key, single_transition)
    assert_proportional(probability(next_state), zero_state)

    mixed_state = jnp.array([[0.4, 0.4, 0.2]])

    next_state, observation = z1r.generate(mixed_state, key, single_transition)
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
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(z1r.normalizing_eigenvector[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = z1r.generate(initial_states, keys, sequence_len)
    assert intermediate_states.shape == (batch_size, z1r.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = z1r.generate(intermediate_states, keys, sequence_len)
    assert final_states.shape == (batch_size, z1r.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


def test_observation_probability_distribution(z1r: HiddenMarkovModel):
    state = jnp.array([0.3, 0.1, 0.6])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))

    state = jnp.array([0.5, 0.3, 0.2])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))


def test_log_observation_probability_distribution(z1r: HiddenMarkovModel):
    log_state = jnp.log(jnp.array([0.3, 0.1, 0.6]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=1e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))

    log_state = jnp.log(jnp.array([0.5, 0.3, 0.2]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=1e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))


def test_probability(z1r: HiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)


def test_log_probability(z1r: HiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))
