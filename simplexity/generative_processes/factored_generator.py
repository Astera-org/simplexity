from typing import Sequence, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess

FactoredState = tuple[jax.Array, ...]


class FactoredGenerativeProcess(GenerativeProcess[FactoredState]):
    """A factored generative process composed of multiple independent HMMs/GHMMs.
    
    Each component generates its own token sequence, then the sequences are combined
    via outer product to create composite tokens. For example:
    - Factor 1 generates: [0, 1, 1, 0]
    - Factor 2 generates: [3, 2, 1, 2]  
    - Combined tuples: [(0,3), (1,2), (1,1), (0,2)]
    - Output tokens: [A, B, C, D] (where each unique tuple maps to a token)
    """
    
    components: Sequence[GenerativeProcess]
    _vocab_size: int
    
    def __init__(self, components: Sequence[GenerativeProcess]):
        """Initialize factored generator with component processes.
        
        Args:
            components: List of HMMs/GHMMs that will generate token sequences independently
        """
        if len(components) == 0:
            raise ValueError("Must provide at least one component process")
            
        self.components = components
        # Vocab size is product of all component vocab sizes
        self._vocab_size = 1
        for component in components:
            self._vocab_size *= component.vocab_size
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (product of component vocab sizes)."""
        return self._vocab_size
    
    @property
    def initial_state(self) -> FactoredState:
        """Initial state is tuple of component initial states."""
        return tuple(component.initial_state for component in self.components)
    
    def _tuple_to_token(self, token_tuple: tuple[int, ...]) -> int:
        """Convert tuple of component tokens to single composite token.
        
        Uses base conversion: for components with vocab sizes [V1, V2, V3],
        tuple (t1, t2, t3) -> t1 * V2 * V3 + t2 * V3 + t3
        """
        token = 0
        multiplier = 1
        
        # Process in reverse order for correct base conversion
        for i in reversed(range(len(token_tuple))):
            token += token_tuple[i] * multiplier
            multiplier *= self.components[i].vocab_size
            
        return token
    
    def _token_to_tuple(self, token: int) -> tuple[int, ...]:
        """Convert composite token back to tuple of component tokens."""
        result = []
        remaining = token
        
        # Process in reverse order
        for i in reversed(range(len(self.components))):
            component_vocab_size = self.components[i].vocab_size
            result.append(remaining % component_vocab_size)
            remaining //= component_vocab_size
            
        return tuple(reversed(result))
    
    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: chex.PRNGKey) -> jax.Array:
        """Emit observation by having each component emit, then combining via outer product."""
        keys = jax.random.split(key, len(self.components))
        component_tokens = []
        
        for i, (component, component_state, component_key) in enumerate(zip(self.components, state, keys)):
            token = component.emit_observation(component_state, component_key)
            component_tokens.append(token)
        
        # Convert list of component tokens to tuple, then to composite token
        token_tuple = tuple(component_tokens)
        return self._tuple_to_token(token_tuple)
    
    @eqx.filter_jit  
    def transition_states(self, state: FactoredState, obs: chex.Array) -> FactoredState:
        """Transition each component state independently based on its part of the observation."""
        # Convert composite observation back to component observations
        component_obs_tuple = self._token_to_tuple(obs)
        
        new_component_states = []
        for component, component_state, component_obs in zip(self.components, state, component_obs_tuple):
            new_state = component.transition_states(component_state, component_obs)
            new_component_states.append(new_state)
            
        return tuple(new_component_states)
    
    # Placeholder implementations - will be completed in Step 4
    def observation_probability_distribution(self, state: FactoredState) -> jax.Array:
        raise NotImplementedError("Will be implemented in Step 4")
    
    def log_observation_probability_distribution(self, log_belief_state: FactoredState) -> jax.Array:
        raise NotImplementedError("Will be implemented in Step 4")
    
    def probability(self, observations: jax.Array) -> jax.Array:
        raise NotImplementedError("Will be implemented in Step 4")
    
    def log_probability(self, observations: jax.Array) -> jax.Array:
        raise NotImplementedError("Will be implemented in Step 4")