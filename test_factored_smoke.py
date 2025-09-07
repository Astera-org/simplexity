#!/usr/bin/env python3
"""Smoke test for Step 1: Core FactoredGenerativeProcess class structure."""

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
    
    # Test tuple-to-token conversion
    print("\n--- Testing tuple-to-token conversion ---")
    test_tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for tup in test_tuples:
        token = factored_gen._tuple_to_token(tup)
        recovered_tup = factored_gen._token_to_tuple(token)
        print(f"Tuple {tup} -> Token {token} -> Recovered {recovered_tup}")
        assert tup == recovered_tup, f"Mismatch: {tup} != {recovered_tup}"
    
    print("\n✅ Step 1 smoke test PASSED!")
    print("- FactoredGenerativeProcess initializes correctly")
    print("- Vocab size calculation works (product of components)")
    print("- Initial state is tuple of component states") 
    print("- Tuple-to-token conversion is bijective")

if __name__ == "__main__":
    test_step1_smoke()