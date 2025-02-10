import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import zero_one_random


@pytest.fixture
def z1r() -> HiddenMarkovModel:
    transition_matrices = zero_one_random()
    return HiddenMarkovModel(transition_matrices)

def test_properties(z1r: HiddenMarkovModel):
    assert z1r.num_observations == 2
    assert z1r.num_states == 3
    stationary_distribution = jnp.ones(3) / 3
    chex.assert_trees_all_close(z1r.right_stationary_distribution, stationary_distribution)
    chex.assert_trees_all_close(z1r.left_stationary_distribution, stationary_distribution)

def test_transition(z1r: HiddenMarkovModel):
    zero_state = jnp.array([1, 0, 0])
    one_state = jnp.array([0, 1, 0])
    random_state = jnp.array([0, 0, 1])

    key = jax.random.PRNGKey(0)
    
    next_state, observation = z1r.transition(zero_state, key)
    chex.assert_trees_all_equal(next_state, one_state)
    assert observation == jnp.array(0)
    
    next_state, observation = z1r.transition(one_state, key)
    chex.assert_trees_all_equal(next_state, random_state)
    assert observation == jnp.array(1)

    next_state, observation = z1r.transition(random_state, key)
    chex.assert_trees_all_equal(next_state, zero_state)

    mixed_state = jnp.array([0.4, 0.4, 0.2])
    
    next_state, observation = z1r.transition(mixed_state, key)
    # P(next=0 | obs=x) = P(prev=2 | obs=x)
    # P(next=1 | obs=x) = P(prev=0 | obs=x)
    # P(next=2 | obs=x) = P(prev=1 | obs=x)
    if observation == 0:
        # P(obs=0 | prev=2) * P(prev=2) = 0.5 * 0.2 = 0.1
        # P(obs=0 | prev=0) * P(prev=0) = 1.0 * 0.4 = 0.4
        # P(obs=0 | prev=1) * P(prev=1) = 0.0 * 0.4 = 0.0
        next_mixed_state = jnp.array([0.2, 0.8, 0.0])
    else:
        # P(obs=1 | prev=2) * P(prev=2) = 0.5 * 0.2 = 0.1
        # P(obs=1 | prev=0) * P(prev=0) = 0.0 * 0.4 = 0.0
        # P(obs=1 | prev=1) * P(prev=1) = 1.0 * 0.4 = 0.4
        next_mixed_state = jnp.array([0.2, 0.0, 0.8])
    chex.assert_trees_all_equal(next_state, next_mixed_state)

def test_generate(z1r: HiddenMarkovModel):
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(z1r.right_stationary_distribution[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = z1r.generate(initial_states, keys, sequence_len)
    assert intermediate_states.shape == (batch_size, z1r.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = z1r.generate(intermediate_states, keys, sequence_len)
    assert final_states.shape == (batch_size, z1r.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


def test_probability(z1r: HiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))
