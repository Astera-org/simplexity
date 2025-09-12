import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_factored_generator, build_factored_hmm_generator
from simplexity.generative_processes.factored_generator import FactoredGenerativeProcess
from simplexity.generative_processes.generator import batch_state, generate_data_batch
from tests.assertions import assert_proportional


@pytest.fixture
def three_component_factored_gen() -> FactoredGenerativeProcess:
    """3-component HMM factored generator: vocab_size = 2 * 3 * 2 = 12"""
    return build_factored_hmm_generator(
        [
            {"process_name": "zero_one_random", "p": 0.6},  # vocab=2
            {"process_name": "mess3", "x": 0.3, "a": 0.7},  # vocab=3
            {"process_name": "zero_one_random", "p": 0.4},  # vocab=2
        ]
    )


@pytest.fixture
def mixed_three_component_gen() -> FactoredGenerativeProcess:
    """3-component mixed HMM/GHMM factored generator: vocab_size = 3 * 4 * 2 = 24"""
    return build_factored_generator(
        [
            {"process_name": "mess3", "x": 0.5, "a": 0.8},  # vocab=3 (HMM)
            {"process_name": "tom_quantum", "alpha": 0.3, "beta": 0.7},  # vocab=4 (GHMM)
            {"process_name": "zero_one_random", "p": 0.9},  # vocab=2 (HMM)
        ],
        component_types=["hmm", "ghmm", "hmm"],
    )


def test_properties(three_component_factored_gen: FactoredGenerativeProcess):
    """Test basic properties of factored generator."""
    assert three_component_factored_gen.vocab_size == 12  # 2 * 3 * 2

    # Test initial_state is tuple of component states
    initial_state = three_component_factored_gen.initial_state
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 3

    # Each component state should be a JAX array
    for component_state in initial_state:
        assert isinstance(component_state, jax.Array)


@pytest.mark.parametrize(
    ("model_name", "vocab_size"),
    [
        ("three_component_factored_gen", 12),  # 2 * 3 * 2
        ("mixed_three_component_gen", 24),  # 3 * 4 * 2
    ],
)
def test_parametrized_properties(model_name: str, vocab_size: int, request: pytest.FixtureRequest):
    """Test properties across different factored generator configurations."""
    model: FactoredGenerativeProcess = request.getfixturevalue(model_name)
    assert model.vocab_size == vocab_size

    initial_state = model.initial_state
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 3  # All test fixtures have 3 components


def test_token_conversion_bijection(three_component_factored_gen: FactoredGenerativeProcess):
    """Test that _tuple_to_token and _token_to_tuple are perfect inverses."""
    model = three_component_factored_gen

    # Test all possible tokens in vocab
    for token in range(model.vocab_size):
        # Convert token → tuple → token
        token_array = jnp.array(token)
        tuple_result = model._token_to_tuple(token_array)
        reconstructed_token = model._tuple_to_token(tuple_result)

        assert jnp.array_equal(token_array, reconstructed_token), (
            f"Token {token} → tuple {tuple_result} → token {reconstructed_token}"
        )

    # Test that all tuples produce unique tokens
    seen_tokens = set()
    for c1 in range(2):  # component 1 vocab size
        for c2 in range(3):  # component 2 vocab size
            for c3 in range(2):  # component 3 vocab size
                tuple_tokens = (jnp.array(c1), jnp.array(c2), jnp.array(c3))
                token = model._tuple_to_token(tuple_tokens)
                token_int = int(token)

                assert token_int not in seen_tokens, f"Duplicate token {token_int} for tuple {tuple_tokens}"
                assert 0 <= token_int < model.vocab_size, f"Token {token_int} out of range [0, {model.vocab_size})"

                seen_tokens.add(token_int)

    # Verify we generated exactly vocab_size unique tokens
    assert len(seen_tokens) == model.vocab_size


def test_token_conversion_examples(three_component_factored_gen: FactoredGenerativeProcess):
    """Test specific token conversion examples with known results."""
    model = three_component_factored_gen

    # Test (0,0,0) → token 0
    tuple_000 = (jnp.array(0), jnp.array(0), jnp.array(0))
    token_000 = model._tuple_to_token(tuple_000)
    assert jnp.array_equal(token_000, jnp.array(0))

    # Test manual calculation: token = c1*3*2 + c2*2 + c3
    # For tuple (1,2,1): token = 1*6 + 2*2 + 1 = 6 + 4 + 1 = 11
    tuple_121 = (jnp.array(1), jnp.array(2), jnp.array(1))
    token_121 = model._tuple_to_token(tuple_121)
    expected_token_121 = 1 * 3 * 2 + 2 * 2 + 1  # 11
    assert jnp.array_equal(token_121, jnp.array(expected_token_121))

    # Test reverse conversion
    reconstructed_tuple = model._token_to_tuple(jnp.array(expected_token_121))
    assert len(reconstructed_tuple) == 3
    assert jnp.array_equal(reconstructed_tuple[0], jnp.array(1))
    assert jnp.array_equal(reconstructed_tuple[1], jnp.array(2))
    assert jnp.array_equal(reconstructed_tuple[2], jnp.array(1))


@pytest.mark.parametrize("model_name", ["three_component_factored_gen", "mixed_three_component_gen"])
def test_generate(model_name: str, request: pytest.FixtureRequest):
    """Test sequence generation with proper shapes and ranges."""
    model: FactoredGenerativeProcess = request.getfixturevalue(model_name)
    batch_size = 4
    sequence_len = 10

    # Create batch of initial states using PyTree batching
    initial_state = model.initial_state
    initial_states = batch_state(initial_state, batch_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    final_states, observations = model.generate(initial_states, keys, sequence_len, False)

    # Test shapes
    assert isinstance(final_states, tuple)
    assert len(final_states) == len(initial_state)
    for i, (final_component_state, initial_component_state) in enumerate(zip(final_states, initial_state)):
        expected_shape = (batch_size, *initial_component_state.shape)
        assert final_component_state.shape == expected_shape, f"Component {i} final state shape mismatch"

    assert observations.shape == (batch_size, sequence_len)

    # Test token ranges
    assert jnp.all(observations >= 0)
    assert jnp.all(observations < model.vocab_size)


@pytest.mark.parametrize("model_name", ["three_component_factored_gen", "mixed_three_component_gen"])
def test_generate_with_intermediate_states(model_name: str, request: pytest.FixtureRequest):
    """Test generation with intermediate states returned."""
    model: FactoredGenerativeProcess = request.getfixturevalue(model_name)
    batch_size = 4
    sequence_len = 10

    initial_state = model.initial_state
    initial_states = batch_state(initial_state, batch_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    intermediate_states, observations = model.generate(initial_states, keys, sequence_len, True)

    # Test intermediate states shape
    assert isinstance(intermediate_states, tuple)
    assert len(intermediate_states) == len(initial_state)
    for i, (intermediate_component_states, initial_component_state) in enumerate(
        zip(intermediate_states, initial_state)
    ):
        expected_shape = (batch_size, sequence_len, *initial_component_state.shape)
        assert intermediate_component_states.shape == expected_shape, (
            f"Component {i} intermediate states shape mismatch"
        )

    assert observations.shape == (batch_size, sequence_len)


def test_observation_probability_distribution(three_component_factored_gen: FactoredGenerativeProcess):
    """Test observation probability distribution computation."""
    model = three_component_factored_gen
    state = model.initial_state

    obs_probs = model.observation_probability_distribution(state)

    # Test shape and properties
    assert obs_probs.shape == (model.vocab_size,)
    assert jnp.all(obs_probs >= 0), "Probabilities should be non-negative"
    assert jnp.isclose(jnp.sum(obs_probs), 1.0, atol=1e-6), "Probabilities should sum to 1"


def test_log_observation_probability_distribution(three_component_factored_gen: FactoredGenerativeProcess):
    """Test log observation probability distribution computation."""
    model = three_component_factored_gen
    state = model.initial_state
    log_state = tuple(jnp.log(component_state) for component_state in state)

    log_obs_probs = model.log_observation_probability_distribution(log_state)

    # Test shape and properties
    assert log_obs_probs.shape == (model.vocab_size,)
    assert jnp.isclose(jax.nn.logsumexp(log_obs_probs), 0.0, atol=1e-6), (
        "Log probabilities should sum to 1 in linear space"
    )

    # Test consistency with regular probability distribution
    obs_probs = model.observation_probability_distribution(state)
    expected_log_probs = jnp.log(obs_probs)
    chex.assert_trees_all_close(log_obs_probs, expected_log_probs, atol=1e-6)


def test_probability(three_component_factored_gen: FactoredGenerativeProcess):
    """Test sequence probability computation."""
    model = three_component_factored_gen

    # Test with a short sequence
    observations = jnp.array([0, 1, 5, 2, 11])  # Valid tokens for vocab_size=12

    probability = model.probability(observations)

    # Basic properties
    assert jnp.isscalar(probability)
    assert 0 <= probability <= 1, f"Probability {probability} not in [0,1]"


def test_log_probability(three_component_factored_gen: FactoredGenerativeProcess):
    """Test log sequence probability computation."""
    model = three_component_factored_gen

    observations = jnp.array([0, 1, 5, 2, 11])

    log_probability = model.log_probability(observations)
    probability = model.probability(observations)

    # Test consistency
    assert jnp.isclose(log_probability, jnp.log(probability), atol=1e-6), (
        "log_probability should equal log(probability)"
    )

    assert log_probability <= 0, "Log probability should be non-positive"


def test_probability_factorization_property(three_component_factored_gen: FactoredGenerativeProcess):
    """Test that probability correctly implements factorization: P(composite) = ∏P(component_i)."""
    model = three_component_factored_gen

    # Create a simple sequence
    composite_observations = jnp.array([0, 6, 11])  # Tokens that decompose nicely

    # Get composite sequence probability
    composite_prob = model.probability(composite_observations)

    # Manually extract component sequences and compute their probabilities
    component_sequences = []
    for i in range(len(model.components)):
        component_seq = []
        for obs in composite_observations:
            component_tokens = model._token_to_tuple(obs)
            component_seq.append(component_tokens[i])
        component_sequences.append(jnp.array(component_seq))

    # Compute individual component probabilities
    manual_prob = 1.0
    for component, component_seq in zip(model.components, component_sequences):
        component_prob = component.probability(component_seq)
        manual_prob *= component_prob

    # They should be equal (factorization property)
    assert jnp.isclose(composite_prob, manual_prob, rtol=1e-5), (
        f"Factorization failed: {composite_prob} != {manual_prob}"
    )


def test_sequence_generation():
    """Test sequence generation functionality."""
    factored_gen = build_factored_hmm_generator(
        [
            {"process_name": "zero_one_random", "p": 0.6},
            {"process_name": "mess3", "x": 0.3, "a": 0.7},
            {"process_name": "zero_one_random", "p": 0.4},
        ]
    )

    batch_size = 10
    sequence_len = 10

    # Test sequence generation functionality
    initial_state = factored_gen.initial_state
    gen_states = batch_state(initial_state, batch_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    final_gen_states, observations = factored_gen.generate(gen_states, keys, sequence_len, False)

    # Manually create inputs and labels like generate_data_batch does
    inputs = observations[:, :-1]
    labels = observations[:, 1:]

    # Test shapes
    assert inputs.shape == (batch_size, sequence_len - 1)
    assert labels.shape == (batch_size, sequence_len - 1)

    # Test ranges
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs < factored_gen.vocab_size)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < factored_gen.vocab_size)

    # Test sequence property: labels should be inputs shifted by 1
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])

    # Test final generator states
    assert isinstance(final_gen_states, tuple)
    assert len(final_gen_states) == len(initial_state)


def test_sequence_generation_with_bos_token():
    """Test sequence generation with BOS token behavior."""
    factored_gen = build_factored_hmm_generator(
        [{"process_name": "zero_one_random", "p": 0.5}, {"process_name": "mess3", "x": 0.4, "a": 0.6}]
    )  # vocab_size = 2 * 3 = 6

    batch_size = 10
    sequence_len = 10
    bos_token = factored_gen.vocab_size  # 6

    initial_state = factored_gen.initial_state
    gen_states = batch_state(initial_state, batch_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    final_gen_states, observations = factored_gen.generate(gen_states, keys, sequence_len, False)

    # Manually simulate BOS token behavior
    tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), observations], axis=1)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]

    # With BOS token, sequences are longer
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)

    # First input should be BOS token
    assert jnp.all(inputs[:, 0] == bos_token)

    # Rest should be in vocab range
    assert jnp.all(inputs[:, 1:] < bos_token)
    assert jnp.all(labels < bos_token)

    # Sequence shift property
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])


def test_sequence_generation_with_eos_token():
    """Test sequence generation with EOS token behavior."""
    factored_gen = build_factored_hmm_generator(
        [{"process_name": "zero_one_random", "p": 0.7}, {"process_name": "zero_one_random", "p": 0.3}]
    )  # vocab_size = 2 * 2 = 4

    batch_size = 10
    sequence_len = 10
    eos_token = factored_gen.vocab_size  # 4

    initial_state = factored_gen.initial_state
    gen_states = batch_state(initial_state, batch_size)

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    final_gen_states, observations = factored_gen.generate(gen_states, keys, sequence_len, False)

    # Manually simulate EOS token behavior
    tokens = jnp.concatenate([observations, jnp.full((batch_size, 1), eos_token)], axis=1)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]

    # With EOS token, sequences are longer
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)

    # Inputs should be in vocab range
    assert jnp.all(inputs < eos_token)

    # Last label should be EOS token
    assert jnp.all(labels[:, -1] == eos_token)

    # Other labels should be in vocab range
    assert jnp.all(labels[:, :-1] < eos_token)

    # Sequence shift property
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
