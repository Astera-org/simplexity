import functools
from collections.abc import Sequence

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
    vocab_sizes: jax.Array  # [V1, V2, ..., VF]
    radix_multipliers: jax.Array  # [V2*V3*...*VF, V3*...*VF, ..., VF, 1]

    def __init__(self, components: Sequence[GenerativeProcess]):
        """Initialize factored generator with component processes.

        Args:
            components: List of HMMs/GHMMs that will generate token sequences independently
        """
        if len(components) == 0:
            raise ValueError("Must provide at least one component process")

        self.components = components

        # Precompute vocab sizes and radix multipliers for vectorized operations
        component_vocab_sizes = [component.vocab_size for component in components]
        self.vocab_sizes = jnp.array(component_vocab_sizes)

        # Compute radix multipliers: [V2*V3*...*VF, V3*...*VF, ..., VF, 1]
        # For vocab sizes [V1, V2, V3], radix should be [V2*V3, V3, 1]
        radix_multipliers = []
        for i in range(len(component_vocab_sizes)):
            # For position i, multiply all vocab sizes from i+1 onwards
            multiplier = 1
            for j in range(i + 1, len(component_vocab_sizes)):
                multiplier *= component_vocab_sizes[j]
            radix_multipliers.append(multiplier)
        self.radix_multipliers = jnp.array(radix_multipliers)

        # Vocab size is product of all component vocab sizes
        self._vocab_size = int(jnp.prod(self.vocab_sizes))

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (product of component vocab sizes)."""
        return self._vocab_size

    @property
    def initial_state(self) -> FactoredState:
        """Initial state is tuple of component initial states."""
        return tuple(component.initial_state for component in self.components)

    def _tuple_to_token(self, token_tuple: tuple[jax.Array, ...]) -> jax.Array:
        """Convert tuple of component tokens to single composite token.

        Uses base conversion: for components with vocab sizes [V1, V2, V3],
        tuple (t1, t2, t3) -> t1 * V2 * V3 + t2 * V3 + t3
        """
        token = jnp.array(0)
        multiplier = jnp.array(1)

        # Process in reverse order for correct base conversion
        for i in reversed(range(len(token_tuple))):
            token += token_tuple[i] * multiplier
            multiplier *= self.components[i].vocab_size

        return token

    def _token_to_tuple(self, token: chex.Array) -> tuple[jax.Array, ...]:
        """Convert composite token back to tuple of component tokens."""
        result = []
        remaining = jnp.array(token)  # Ensure it's a JAX array

        # Process in reverse order
        for i in reversed(range(len(self.components))):
            component_vocab_size = jnp.array(self.components[i].vocab_size)
            component_token = remaining % component_vocab_size
            result.append(component_token)
            remaining = remaining // component_vocab_size

        return tuple(reversed(result))

    def _extract_factors_vectorized(self, tokens: chex.Array) -> jax.Array:
        """Vectorized extraction of component factors from composite tokens.

        Args:
            tokens: Array of composite tokens, shape [T] or [T,]

        Returns:
            factors: Array of component factors, shape [T, F] where F is num_components
        """
        # Ensure tokens is at least 1D
        tokens = jnp.atleast_1d(tokens)

        # Vectorized base conversion: tokens[i, None] broadcasts to [T, 1]
        # radix_multipliers[None, :] broadcasts to [1, F]
        # Result is [T, F] with factors[t, f] = factor f for token t
        factors = (tokens[:, None] // self.radix_multipliers[None, :]) % self.vocab_sizes[None, :]
        return factors

    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: chex.PRNGKey) -> jax.Array:
        """Emit observation by having each component emit, then combining via outer product."""
        keys = jax.random.split(key, len(self.components))

        # Emit tokens from each component
        component_tokens = []
        for component, component_state, component_key in zip(self.components, state, keys, strict=True):
            token = component.emit_observation(component_state, component_key)
            component_tokens.append(token)

        # Convert to tuple and then to composite token
        token_tuple = tuple(component_tokens)
        return self._tuple_to_token(token_tuple)

    @eqx.filter_jit
    def transition_states(self, state: FactoredState, obs: chex.Array) -> FactoredState:
        """Transition each component state independently based on its part of the observation."""
        # Convert composite observation back to component observations
        component_obs_tuple = self._token_to_tuple(obs)

        new_component_states = []
        for component, component_state, component_obs in zip(self.components, state, component_obs_tuple, strict=True):
            new_state = component.transition_states(component_state, component_obs)
            new_component_states.append(new_state)

        return tuple(new_component_states)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: FactoredState) -> jax.Array:
        """Compute probability distribution over composite observations using Kronecker product.

        For each possible composite token (i,j,...), compute P(composite_token | factored_state)
        = P(token_i | component_state_1) * P(token_j | component_state_2) * ...

        Uses vectorized Kronecker product instead of explicit loops over vocab_size.
        """
        # Get probability distributions for each component
        component_probs = []
        for component, component_state in zip(self.components, state, strict=True):
            component_prob_dist = component.observation_probability_distribution(component_state)
            component_probs.append(component_prob_dist)

        # Compute outer product via Kronecker product
        # functools.reduce applies jnp.kron sequentially: kron(kron(p1, p2), p3), etc.
        composite_probs = functools.reduce(jnp.kron, component_probs)

        return composite_probs

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: FactoredState) -> jax.Array:
        """Compute log probability distribution over composite observations."""
        # Convert from log space to regular space, compute distribution, convert back
        # This is not the most numerically stable approach, but matches existing pattern in GHMM
        regular_state = tuple(jnp.exp(component_log_state) for component_log_state in log_belief_state)
        obs_prob_dist = self.observation_probability_distribution(regular_state)
        return jnp.log(obs_prob_dist)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute probability of generating observation sequence using vectorized operations.

        Since components are independent, we can compute this by running each component
        independently and multiplying their probabilities.
        """
        # Vectorized extraction of all component factors at once
        # factors shape: [T, F] where T=sequence_length, F=num_components
        factors = self._extract_factors_vectorized(observations)

        # Compute probability for each component independently and multiply
        total_prob = jnp.array(1.0)
        for i, component in enumerate(self.components):
            component_seq = factors[:, i]  # Extract sequence for component i
            component_prob = component.probability(component_seq)
            total_prob *= component_prob

        return total_prob

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute log probability of generating observation sequence using vectorized operations."""
        # Vectorized extraction of all component factors at once
        # factors shape: [T, F] where T=sequence_length, F=num_components
        factors = self._extract_factors_vectorized(observations)

        # Compute log probability for each component independently and sum
        total_log_prob = jnp.array(0.0)
        for i, component in enumerate(self.components):
            component_seq = factors[:, i]  # Extract sequence for component i
            component_log_prob = component.log_probability(component_seq)
            total_log_prob += component_log_prob

        return total_log_prob
