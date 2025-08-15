import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_hidden_markov_model, build_product_generator
from simplexity.generative_processes.product_generator import ProductGenerator, ProductState


def test_product_generator_initialization():
    """Test that ProductGenerator initializes correctly."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    gen2 = build_hidden_markov_model("zero_one_random", p=0.3)
    
    product_gen = ProductGenerator([gen1, gen2])
    
    assert product_gen.vocab_size == gen1.vocab_size * gen2.vocab_size
    assert len(product_gen.component_generators) == 2
    assert isinstance(product_gen.initial_state, ProductState)
    assert len(product_gen.initial_state.factored_states) == 2


def test_product_generator_requires_multiple_components():
    """Test that ProductGenerator requires at least 2 components."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    
    with pytest.raises(ValueError, match="at least 2 component generators"):
        ProductGenerator([gen1])


def test_token_encoding_decoding():
    """Test that token encoding and decoding are inverses."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)  # vocab_size = 2
    gen2 = build_hidden_markov_model("even_ones")  # vocab_size = 2
    
    product_gen = ProductGenerator([gen1, gen2])
    
    # Test all possible token combinations
    for token1 in range(gen1.vocab_size):
        for token2 in range(gen2.vocab_size):
            component_tokens = [token1, token2]
            product_token = product_gen._encode_token(component_tokens)
            decoded_tokens = product_gen._decode_token(product_token)
            
            assert decoded_tokens == component_tokens
            assert 0 <= product_token < product_gen.vocab_size


def test_emit_observation():
    """Test that emit_observation produces valid tokens."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    gen2 = build_hidden_markov_model("zero_one_random", p=0.3)
    
    product_gen = ProductGenerator([gen1, gen2])
    state = product_gen.initial_state
    key = jax.random.PRNGKey(0)
    
    obs = product_gen.emit_observation(state, key)
    
    assert 0 <= obs < product_gen.vocab_size
    assert obs.shape == ()


def test_transition_states():
    """Test state transitions."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    gen2 = build_hidden_markov_model("zero_one_random", p=0.3)
    
    product_gen = ProductGenerator([gen1, gen2])
    state = product_gen.initial_state
    
    # Test transition with each possible observation
    for obs in range(product_gen.vocab_size):
        new_state = product_gen.transition_states(state, obs)
        
        assert isinstance(new_state, ProductState)
        assert len(new_state.factored_states) == 2
        
        # Check that states are normalized
        for component_state in new_state.factored_states:
            assert jnp.allclose(jnp.sum(component_state), 1.0, atol=1e-6)


def test_observation_probability_distribution():
    """Test that observation probabilities sum to 1."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    gen2 = build_hidden_markov_model("zero_one_random", p=0.3)
    
    product_gen = ProductGenerator([gen1, gen2])
    state = product_gen.initial_state
    
    probs = product_gen.observation_probability_distribution(state)
    
    assert probs.shape == (product_gen.vocab_size,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
    assert jnp.all(probs >= 0)
    assert jnp.all(probs <= 1)


def test_generate_sequences():
    """Test sequence generation."""
    gen1 = build_hidden_markov_model("zero_one_random", p=0.5)
    gen2 = build_hidden_markov_model("zero_one_random", p=0.3)
    
    product_gen = ProductGenerator([gen1, gen2])
    state = product_gen.initial_state
    key = jax.random.PRNGKey(0)
    sequence_len = 10
    
    # Test without returning all states
    final_state, observations = product_gen.generate(state, key, sequence_len, return_all_states=False)
    
    assert observations.shape == (sequence_len,)
    assert jnp.all(observations >= 0)
    assert jnp.all(observations < product_gen.vocab_size)
    assert isinstance(final_state, ProductState)
    
    # Test with returning all states
    all_states, observations = product_gen.generate(state, key, sequence_len, return_all_states=True)
    
    assert observations.shape == (sequence_len,)
    assert isinstance(all_states, ProductState)
    assert len(all_states.factored_states) == 2
    assert all_states.factored_states[0].shape == (sequence_len, gen1.initial_state.shape[0])
    assert all_states.factored_states[1].shape == (sequence_len, gen2.initial_state.shape[0])


def test_probability_computation():
    """Test probability computation for a sequence."""
    gen1 = build_hidden_markov_model("zero_one_random", p=1.0)  # Always outputs 1
    gen2 = build_hidden_markov_model("zero_one_random", p=0.0)  # Always outputs 0
    
    product_gen = ProductGenerator([gen1, gen2])
    
    # Token encoding: [1, 0] -> 1*1 + 0 = 1
    observations = jnp.array([1, 1, 1])
    
    prob = product_gen.probability(observations)
    
    # Since both generators are deterministic, probability should be 1
    assert jnp.allclose(prob, 1.0, atol=1e-6)
    
    # Try an impossible sequence
    observations = jnp.array([0, 0, 0])  # This would be [0, 0] in component tokens
    prob = product_gen.probability(observations)
    
    # This should have probability 0 since gen1 never outputs 0
    assert jnp.allclose(prob, 0.0, atol=1e-6)


def test_build_product_generator():
    """Test the builder function."""
    configs = [
        {'type': 'hmm', 'process_name': 'zero_one_random', 'p': 0.5},
        {'type': 'hmm', 'process_name': 'even_ones'}
    ]
    
    product_gen = build_product_generator(configs)
    
    assert isinstance(product_gen, ProductGenerator)
    assert len(product_gen.component_generators) == 2
    assert product_gen.vocab_size == 4  # 2 * 2


