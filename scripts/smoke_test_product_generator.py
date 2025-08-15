#!/usr/bin/env python3
"""Smoke test for ProductGenerator showing cartesian products and belief states."""

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.generative_processes.builder import build_product_generator


def format_state(state, name="State"):
    """Format a state vector for display."""
    state_str = ", ".join([f"{val:.3f}" for val in state])
    return f"{name}: [{state_str}]"


def decode_token_to_binary(token, num_components):
    """Decode a product token to binary representation."""
    binary = []
    for _ in range(num_components):
        binary.append(token % 2)
        token = token // 2
    return binary


def main():
    print("=" * 80)
    print("PRODUCT GENERATOR SMOKE TEST")
    print("=" * 80)
    print()
    
    # Create component generators
    print("Creating component generators:")
    print("  1. Mess3 (x=0.15, a=0.6) - 3-state, 3-token HMM")
    print("  2. Tom Quantum (alpha=1.0, beta=1.0) - 3-state, 4-token GHMM")
    print()
    
    configs = [
        {'type': 'hmm', 'process_name': 'mess3', 'x': 0.15, 'a': 0.6},
        {'type': 'ghmm', 'process_name': 'tom_quantum', 'alpha': 1.0, 'beta': 1.0}
    ]
    
    product_gen = build_product_generator(configs)
    
    print(f"Product Generator Info:")
    print(f"  Component vocab sizes: {[gen.vocab_size for gen in product_gen.component_generators]}")
    print(f"  Product vocab size: {product_gen.vocab_size}")
    print(f"  Token mapping (product -> [comp1, comp2]):")
    for i in range(product_gen.vocab_size):
        comp_tokens = product_gen._decode_token(i)
        print(f"    {i} -> {comp_tokens}")
    print()
    
    # Generate sequences
    print("Generating sequences:")
    print("-" * 40)
    
    key = jax.random.PRNGKey(42)
    num_sequences = 3
    sequence_length = 15
    
    for seq_idx in range(num_sequences):
        key, subkey = jax.random.split(key)
        
        print(f"\nSequence {seq_idx + 1}:")
        print("-" * 20)
        
        # Generate with all states returned
        state = product_gen.initial_state
        all_states, observations = product_gen.generate(
            state, subkey, sequence_length, return_all_states=True
        )
        
        # Display initial state
        print("\nInitial State:")
        print(f"  Component 1 (mess3): {format_state(state.factored_states[0])}")
        print(f"  Component 2 (tom_quantum): {format_state(state.factored_states[1])}")
        print()
        
        # Display generated sequence
        print("Generated tokens and their decomposition:")
        print("  Step | Product | [Comp1, Comp2] | Comp1 State | Comp2 State")
        print("  " + "-" * 60)
        
        for t in range(sequence_length):
            product_token = observations[t]
            comp_tokens = product_gen._decode_token(product_token)
            
            # Get belief states at this timestep
            comp1_state = all_states.factored_states[0][t]
            comp2_state = all_states.factored_states[1][t]
            
            comp1_str = ", ".join([f"{val:.2f}" for val in comp1_state])
            comp2_str = ", ".join([f"{val:.2f}" for val in comp2_state])
            
            print(f"  {t:4d} | {product_token:7d} | {comp_tokens:14} | [{comp1_str}] | [{comp2_str}]")
        
        # Compute and display sequence probability
        prob = product_gen.probability(observations)
        log_prob = product_gen.log_probability(observations)
        print(f"\nSequence probability: {prob:.6e}")
        print(f"Log probability: {log_prob:.6f}")
        
        # Show observation probability distribution at final state
        from simplexity.generative_processes.product_generator import ProductState
        final_factored_states = [
            all_states.factored_states[0][-1],
            all_states.factored_states[1][-1]
        ]
        final_state = ProductState(factored_states=final_factored_states)
        
        obs_probs = product_gen.observation_probability_distribution(final_state)
        print("\nObservation probabilities at final state:")
        for i in range(product_gen.vocab_size):
            comp_tokens = product_gen._decode_token(i)
            print(f"  Token {i} {comp_tokens}: {obs_probs[i]:.4f}")
    
    print("\n" + "=" * 80)
    print("CARTESIAN PRODUCT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create a simpler product for clear demonstration
    print("Creating simpler generators:")
    print("  1. Zero-One Random (p=0.8) - vocab_size=2")
    print("  2. Mess3 (x=0.1, a=0.5) - vocab_size=3")
    print()
    
    simple_configs = [
        {'type': 'hmm', 'process_name': 'zero_one_random', 'p': 0.8},
        {'type': 'hmm', 'process_name': 'mess3', 'x': 0.1, 'a': 0.5}
    ]
    
    simple_gen = build_product_generator(simple_configs)
    
    print(f"Product vocab size: {simple_gen.vocab_size} (2×3=6)")
    print("\nToken mapping:")
    for i in range(simple_gen.vocab_size):
        comp_tokens = simple_gen._decode_token(i)
        print(f"  Token {i}: [comp1={comp_tokens[0]}, comp2={comp_tokens[1]}]")
    
    key, subkey = jax.random.split(key)
    state = simple_gen.initial_state
    _, simple_obs = simple_gen.generate(state, subkey, 10, return_all_states=False)
    
    print("\nGenerated sequence:")
    for t, token in enumerate(simple_obs):
        comp_tokens = simple_gen._decode_token(token)
        print(f"  Step {t}: Token {token} = {comp_tokens}")
    
    # Verify the product vocab size is correct
    print("\n" + "=" * 80)
    print("VERIFYING PRODUCT DIMENSIONS")
    print("=" * 80)
    print()
    
    print("Main product generator (mess3 × tom_quantum):")
    print(f"  Mess3: vocab_size = 3")
    print(f"  Tom Quantum: vocab_size = 4")
    print(f"  Product: vocab_size = {product_gen.vocab_size} (3×4=12)")
    print(f"  Note: 3×4=12, NOT 3+4=7")
    print()
    
    # Show all token mappings for the main generator
    print("Full token mapping for mess3 × tom_quantum:")
    for i in range(min(product_gen.vocab_size, 12)):  # Show first 12
        comp_tokens = product_gen._decode_token(i)
        print(f"  Token {i:2d}: [mess3={comp_tokens[0]}, tom_quantum={comp_tokens[1]}]")
    
    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()