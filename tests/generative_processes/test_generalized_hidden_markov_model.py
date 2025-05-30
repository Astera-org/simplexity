import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_generalized_hidden_markov_model
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from tests.assertions import assert_proportional


@pytest.fixture
def z1r() -> GeneralizedHiddenMarkovModel:
    return build_generalized_hidden_markov_model("zero_one_random", p=0.5)


@pytest.fixture
def fanizza_model() -> GeneralizedHiddenMarkovModel:
    return build_generalized_hidden_markov_model("fanizza", alpha=2000, lamb=0.49)


@pytest.mark.parametrize(("model_name", "vocab_size", "num_states"), [("z1r", 2, 3), ("fanizza_model", 2, 4)])
def test_properties(model_name: str, vocab_size: int, num_states: int, request: pytest.FixtureRequest):
    model: GeneralizedHiddenMarkovModel = request.getfixturevalue(model_name)
    assert model.vocab_size == vocab_size
    assert model.num_states == num_states


def test_normalize_belief_state(z1r: GeneralizedHiddenMarkovModel):
    state = jnp.array([2, 5, 1])
    belief_state = z1r.normalize_belief_state(state)
    chex.assert_trees_all_close(belief_state, jnp.array([0.25, 0.625, 0.125]))

    state = jnp.array([0, 0, 0])
    belief_state = z1r.normalize_belief_state(state)
    assert jnp.all(jnp.isnan(belief_state))


def test_normalize_log_belief_state(z1r: GeneralizedHiddenMarkovModel):
    state = jnp.log(jnp.array([2, 5, 1]))
    log_belief_state = z1r.normalize_log_belief_state(state)
    chex.assert_trees_all_close(log_belief_state, jnp.log(jnp.array([0.25, 0.625, 0.125])))

    log_belief_state = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
    log_belief_state = z1r.normalize_log_belief_state(log_belief_state)
    assert jnp.all(jnp.isnan(log_belief_state))


def test_hmm_single_transition(z1r: GeneralizedHiddenMarkovModel):
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
    state = jnp.array([0.3, 0.1, 0.6])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))

    state = jnp.array([0.5, 0.3, 0.2])
    obs_probs = z1r.observation_probability_distribution(state)
    chex.assert_trees_all_close(obs_probs, jnp.array([0.6, 0.4]))


def test_ghmm_observation_probability_distribution(fanizza_model: GeneralizedHiddenMarkovModel):
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
    log_belief_state = jnp.log(jnp.array([0.3, 0.1, 0.6]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=2e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))

    log_belief_state = jnp.log(jnp.array([0.5, 0.3, 0.2]))
    log_obs_probs = z1r.log_observation_probability_distribution(log_belief_state)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0, atol=2e-7)
    chex.assert_trees_all_close(log_obs_probs, jnp.log(jnp.array([0.6, 0.4])))


def test_hmm_probability(z1r: GeneralizedHiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    probability = z1r.probability(observations)
    assert jnp.isclose(probability, expected_probability)


def test_ghmm_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.vocab_size)

    probability = fanizza_model.probability(observations)
    assert 0 <= probability <= 1


def test_hmm_log_probability(z1r: GeneralizedHiddenMarkovModel):
    observations = jnp.array([1, 0, 0, 1, 1, 0])
    expected_probability = 1 / 12

    log_probability = z1r.log_probability(observations)
    assert jnp.isclose(log_probability, jnp.log(expected_probability))


def test_ghmm_log_probability(fanizza_model: GeneralizedHiddenMarkovModel):
    key = jax.random.PRNGKey(0)
    observations = jax.random.randint(key, (10,), 0, fanizza_model.vocab_size)

    log_probability = fanizza_model.log_probability(observations)
    assert log_probability <= 0
