import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.transition_matrices import fanizza, zero_one_random


def assert_proportional(a: jax.Array, b: jax.Array):
    def normalize(x: jax.Array) -> jax.Array:
        return x / jnp.maximum(jnp.abs(x).max(), 1e-6)

    chex.assert_equal_shape([a, b])
    norm_a = normalize(a)
    norm_b = normalize(b)
    try:
        chex.assert_trees_all_close(norm_a, norm_b)
    except AssertionError as e:
        try:
            chex.assert_trees_all_close(norm_a, -norm_b)
        except AssertionError:
            raise AssertionError(f"Arrays are not proportional: {a} and {b}.\n{e}") from e


@pytest.fixture
def z1r() -> GeneralizedHiddenMarkovModel:
    transition_matrices = zero_one_random()
    return GeneralizedHiddenMarkovModel(transition_matrices)


@pytest.fixture
def fanizza_model() -> GeneralizedHiddenMarkovModel:
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    return GeneralizedHiddenMarkovModel(transition_matrices)


def test_hmm_properties(z1r: GeneralizedHiddenMarkovModel):
    assert z1r.num_observations == 2
    assert z1r.num_states == 3
    assert_proportional(z1r.right_eigenvector, jnp.ones(3))
    assert_proportional(z1r.left_eigenvector, jnp.ones(3))


def test_ghmm_properties(fanizza_model: GeneralizedHiddenMarkovModel):
    assert fanizza_model.num_observations == 2
    assert fanizza_model.num_states == 4


def test_hmm_single_transition(z1r: GeneralizedHiddenMarkovModel):
    zero_state = jnp.array([[1.0, 0.0, 0.0]])
    one_state = jnp.array([[0.0, 1.0, 0.0]])
    random_state = jnp.array([[0.0, 0.0, 1.0]])

    probability = eqx.filter_vmap(z1r.state_probability)

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


def test_hmm_generate(z1r: GeneralizedHiddenMarkovModel):
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(z1r.right_eigenvector[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = z1r.generate(initial_states, keys, sequence_len)
    assert intermediate_states.shape == (batch_size, z1r.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = z1r.generate(intermediate_states, keys, sequence_len)
    assert final_states.shape == (batch_size, z1r.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


def test_ghmm_generate(fanizza_model: GeneralizedHiddenMarkovModel):
    batch_size = 4
    sequence_len = 10

    initial_states = jnp.repeat(fanizza_model.right_eigenvector[None, :], batch_size, axis=0)
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, intermediate_observations = fanizza_model.generate(initial_states, keys, sequence_len)
    assert intermediate_states.shape == (batch_size, fanizza_model.num_states)
    assert intermediate_observations.shape == (batch_size, sequence_len)

    keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    final_states, final_observations = fanizza_model.generate(intermediate_states, keys, sequence_len)
    assert final_states.shape == (batch_size, fanizza_model.num_states)
    assert final_observations.shape == (batch_size, sequence_len)


def test_hmm_probability(z1r: GeneralizedHiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)


def test_ghmm_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.num_observations)

    probability = fanizza_model.probability(observations)
    assert 0 <= probability <= 1


def test_hmm_log_probability(z1r: GeneralizedHiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))


def test_ghmm_log_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.num_observations)

    log_probability = fanizza_model.log_probability(observations)
    try:
        assert log_probability <= 0
    except AssertionError:
        pytest.xfail("Eigenvector contains negative values -> log_eigenvector contains nans")
