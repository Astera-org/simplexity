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
