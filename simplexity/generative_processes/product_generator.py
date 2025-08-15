from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


class ProductState(NamedTuple):
    """State for ProductGenerator - stores factored states."""
    factored_states: list[jax.Array]


class ProductGenerator(GenerativeProcess[ProductState]):
    """Generates cartesian products of sequences from multiple component generators."""
    
    component_generators: list[GenerativeProcess]
    component_vocab_sizes: jax.Array
    
    def __init__(self, component_generators: list[GenerativeProcess]):
        if len(component_generators) < 2:
            raise ValueError("ProductGenerator requires at least 2 component generators")
        
        self.component_generators = component_generators
        self.component_vocab_sizes = jnp.array([gen.vocab_size for gen in component_generators])
    
    @property
    def vocab_size(self) -> int:
        return int(jnp.prod(self.component_vocab_sizes))
    
    @property
    def initial_state(self) -> ProductState:
        return ProductState([gen.initial_state for gen in self.component_generators])
    
    def _encode_token(self, component_tokens: list[int]) -> int:
        """Convert list of component tokens to single product token."""
        result = 0
        multiplier = 1
        for token, vocab_size in zip(reversed(component_tokens), reversed(self.component_vocab_sizes)):
            result += token * multiplier
            multiplier *= vocab_size
        return result
    
    def _decode_token(self, product_token: int) -> list[int]:
        """Convert product token back to component tokens."""
        component_tokens = []
        for vocab_size in reversed(self.component_vocab_sizes):
            component_tokens.append(product_token % int(vocab_size))
            product_token //= int(vocab_size)
        return list(reversed(component_tokens))
    
    @eqx.filter_jit
    def emit_observation(self, state: ProductState, key: chex.PRNGKey) -> jax.Array:
        keys = jax.random.split(key, len(self.component_generators))
        
        component_tokens = []
        for gen, gen_state, gen_key in zip(
            self.component_generators, state.factored_states, keys
        ):
            token = gen.emit_observation(gen_state, gen_key)
            component_tokens.append(token)
        
        return self._encode_token(component_tokens)
    
    @eqx.filter_jit
    def transition_states(self, state: ProductState, obs: chex.Array) -> ProductState:
        component_tokens = self._decode_token(obs)
        
        new_states = []
        for gen, gen_state, token in zip(
            self.component_generators, state.factored_states, component_tokens
        ):
            new_states.append(gen.transition_states(gen_state, token))
        
        return ProductState(new_states)
    
    @eqx.filter_jit
    def observation_probability_distribution(self, state: ProductState) -> jax.Array:
        # Get each component's distribution
        component_dists = [
            gen.observation_probability_distribution(gen_state)
            for gen, gen_state in zip(self.component_generators, state.factored_states)
        ]
        
        # Compute kronecker product
        result = component_dists[0]
        for dist in component_dists[1:]:
            result = jnp.kron(result, dist)
        return result
    
    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: ProductState) -> jax.Array:
        # Get each component's log distribution
        component_log_dists = [
            gen.log_observation_probability_distribution(log_state)
            for gen, log_state in zip(self.component_generators, log_belief_state.factored_states)
        ]
        
        # In log space, product becomes sum, but we need to handle the kronecker structure
        # For now, convert to regular space (TODO: proper implementation)
        component_dists = [jnp.exp(log_dist) for log_dist in component_log_dists]
        result = component_dists[0]
        for dist in component_dists[1:]:
            result = jnp.kron(result, dist)
        return jnp.log(result + 1e-10)
    
    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        state = self.initial_state
        prob = 1.0
        
        for obs in observations:
            obs_probs = self.observation_probability_distribution(state)
            prob *= obs_probs[obs]
            state = self.transition_states(state, obs)
        
        return prob
    
    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        state = self.initial_state
        log_prob = 0.0
        
        for obs in observations:
            log_obs_probs = self.log_observation_probability_distribution(state)
            log_prob += log_obs_probs[obs]
            state = self.transition_states(state, obs)
        
        return log_prob