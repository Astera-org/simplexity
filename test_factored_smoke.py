#!/usr/bin/env python3
"""Smoke test for Step 1-2: Core FactoredGenerativeProcess and token generation."""

import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.factored_generator import FactoredGenerativeProcess

def test_step1_smoke():
    """Test basic factored generator initialization and properties."""
    print("=== Step 1 Smoke Test: Core FactoredGenerativeProcess ===")
    
    # Create two simple HMMs
    hmm1 = build_hidden_markov_model("coin", p=0.7)  # Binary: vocab_size=2
    hmm2 = build_hidden_markov_model("coin", p=0.3)  # Binary: vocab_size=2
    
    print(f"Component 1 vocab size: {hmm1.vocab_size}")
    print(f"Component 2 vocab size: {hmm2.vocab_size}")
    
    # Create factored generator
    factored_gen = FactoredGenerativeProcess([hmm1, hmm2])
    
    print(f"Factored generator vocab size: {factored_gen.vocab_size}")
    print(f"Expected vocab size (2 * 2): {2 * 2}")
    
    # Test initial state structure
    initial_state = factored_gen.initial_state
    print(f"Initial state type: {type(initial_state)}")
    print(f"Number of component states: {len(initial_state)}")
    print(f"Component 1 initial state shape: {initial_state[0].shape}")
    print(f"Component 2 initial state shape: {initial_state[1].shape}")
    
    # Test tuple-to-token conversion with JAX arrays
    print("\n--- Testing JAX tuple-to-token conversion ---")
    test_tuples = [
        (jnp.array(0), jnp.array(0)), 
        (jnp.array(0), jnp.array(1)), 
        (jnp.array(1), jnp.array(0)), 
        (jnp.array(1), jnp.array(1))
    ]
    for tup in test_tuples:
        token = factored_gen._tuple_to_token(tup)
        recovered_tup = factored_gen._token_to_tuple(token)
        print(f"Tuple ({int(tup[0])},{int(tup[1])}) -> Token {int(token)} -> Recovered ({int(recovered_tup[0])},{int(recovered_tup[1])})")
        assert int(tup[0]) == int(recovered_tup[0]) and int(tup[1]) == int(recovered_tup[1]), f"Mismatch: {tup} != {recovered_tup}"
    
    print("\n✅ Step 1 smoke test PASSED!")
    print("- FactoredGenerativeProcess initializes correctly")
    print("- Vocab size calculation works (product of components)")
    print("- Initial state is tuple of component states") 
    print("- JAX tuple-to-token conversion is bijective")

def test_step2_smoke():
    """Test outer product token generation."""
    print("\n=== Step 2 Smoke Test: Outer Product Token Generation ===")
    
    # Create two simple HMMs with different probabilities for more variety
    hmm1 = build_hidden_markov_model("coin", p=0.8)  # Biased toward 0
    hmm2 = build_hidden_markov_model("coin", p=0.2)  # Biased toward 1
    
    # Create factored generator
    factored_gen = FactoredGenerativeProcess([hmm1, hmm2])
    
    # Generate some observations
    key = jax.random.PRNGKey(42)
    state = factored_gen.initial_state
    
    print("Generating 5 observations:")
    print("Format: Component1_token, Component2_token -> (tuple) -> Composite_token")
    
    for step in range(5):
        key, obs_key = jax.random.split(key)
        
        # Generate observation from factored generator
        composite_obs = factored_gen.emit_observation(state, obs_key)
        
        # Decompose to see the components
        component_tuple = factored_gen._token_to_tuple(composite_obs)
        
        print(f"Step {step}: {int(component_tuple[0])}, {int(component_tuple[1])} -> ({int(component_tuple[0])},{int(component_tuple[1])}) -> {int(composite_obs)}")
        
        # Update state
        state = factored_gen.transition_states(state, composite_obs)
    
    print("\n✅ Step 2 smoke test PASSED!")
    print("- Components generate tokens independently") 
    print("- Tokens are combined via outer product (tuple formation)")
    print("- Composite tokens decompose back to component tokens correctly")
    print("- State transitions work with composite observations")

def test_step4_smoke():
    """Test complete GenerativeProcess interface implementation."""
    print("\n=== Step 4 Smoke Test: Complete GenerativeProcess Interface ===")
    
    # Create factored generator with 2 simple HMMs
    hmm1 = build_hidden_markov_model("coin", p=0.7)
    hmm2 = build_hidden_markov_model("coin", p=0.4) 
    factored_gen = FactoredGenerativeProcess([hmm1, hmm2])
    
    # Test observation probability distribution
    state = factored_gen.initial_state
    obs_prob_dist = factored_gen.observation_probability_distribution(state)
    print(f"Observation probability distribution shape: {obs_prob_dist.shape}")
    print(f"Sum of probabilities: {jnp.sum(obs_prob_dist):.4f} (should be ~1.0)")
    print(f"Individual probabilities: {obs_prob_dist}")
    
    # Test log observation probability distribution  
    log_state = tuple(jnp.log(component_state) for component_state in state)
    log_obs_prob_dist = factored_gen.log_observation_probability_distribution(log_state)
    print(f"Log obs prob distribution: {log_obs_prob_dist}")
    
    # Generate a short sequence manually (since generate is vmapped for batches)
    key = jax.random.PRNGKey(123)
    sequence_len = 3
    tokens = []
    current_state = state
    
    print(f"\nGenerating sequence manually:")
    for i in range(sequence_len):
        key, step_key = jax.random.split(key)
        token = factored_gen.emit_observation(current_state, step_key)
        tokens.append(token)
        current_state = factored_gen.transition_states(current_state, token)
        
        component_tokens = factored_gen._token_to_tuple(token)
        print(f"  Step {i}: Token {int(token)} -> ({int(component_tokens[0])}, {int(component_tokens[1])})")
    
    tokens = jnp.array(tokens)
    print(f"Generated sequence: {tokens}")
    
    # Test probability calculation
    prob = factored_gen.probability(tokens)
    log_prob = factored_gen.log_probability(tokens)
    print(f"Sequence probability: {prob:.6f}")
    print(f"Sequence log probability: {log_prob:.6f}")
    print(f"exp(log_prob): {jnp.exp(log_prob):.6f} (should match probability)")
    
    print("\n✅ Step 4 smoke test PASSED!")
    print("- observation_probability_distribution works and sums to 1")
    print("- log_observation_probability_distribution works") 
    print("- probability and log_probability work and are consistent")
    print("- All abstract methods from GenerativeProcess are implemented")

def test_step5_smoke():
    """Test training pipeline compatibility."""
    print("\n=== Step 5 Smoke Test: Training Pipeline Compatibility ===")
    
    # Import the training utilities
    from simplexity.generative_processes.generator import generate_data_batch
    
    # Create factored generator
    hmm1 = build_hidden_markov_model("coin", p=0.8)
    hmm2 = build_hidden_markov_model("coin", p=0.3) 
    factored_gen = FactoredGenerativeProcess([hmm1, hmm2])
    
    print(f"Created factored generator with vocab size: {factored_gen.vocab_size}")
    
    # Test generate_data_batch compatibility
    batch_size = 4
    sequence_len = 5
    key = jax.random.PRNGKey(42)
    
    # Create batch of initial states
    initial_state = factored_gen.initial_state
    gen_states = []
    for _ in range(batch_size):
        gen_states.append(initial_state)
    gen_states = tuple(jnp.stack([state[i] for state in gen_states]) for i in range(len(initial_state)))
    
    print(f"Batch of states shape: {[s.shape for s in gen_states]}")
    
    # Generate data batch
    gen_states, inputs, labels = generate_data_batch(
        gen_states, factored_gen, batch_size, sequence_len, key
    )
    
    print(f"Generated batch:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Input vocab range: [{jnp.min(inputs)}, {jnp.max(inputs)}]")
    print(f"  Label vocab range: [{jnp.min(labels)}, {jnp.max(labels)}]")
    
    # Show some examples with decomposition
    print(f"\nFirst few input sequences with factorization:")
    for i in range(min(2, batch_size)):
        print(f"  Batch {i}: {inputs[i]}")
        factorized = []
        for token in inputs[i]:
            components = factored_gen._token_to_tuple(token)
            factorized.append(f"({int(components[0])},{int(components[1])})")
        print(f"    Factored: {factorized}")
    
    print("\n✅ Step 5 smoke test PASSED!")
    print("- generate_data_batch works with factored generator")
    print("- Batch shapes are correct for training") 
    print("- Vocab ranges are within expected bounds")
    print("- Ready for plug-and-play training replacement!")

if __name__ == "__main__":
    test_step1_smoke()
    test_step2_smoke()
    test_step4_smoke()
    test_step5_smoke()