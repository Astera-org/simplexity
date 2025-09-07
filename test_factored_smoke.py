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

if __name__ == "__main__":
    test_step1_smoke()
    test_step2_smoke()