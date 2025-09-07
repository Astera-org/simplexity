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
        remaining = token

        # Process in reverse order
        for i in reversed(range(len(self.components))):
            component_vocab_size = self.components[i].vocab_size
            component_token = remaining % component_vocab_size
            result.append(component_token)
            remaining //= component_vocab_size

        return tuple(reversed(result))

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
        """Compute probability distribution over composite observations.

        For each possible composite token (i,j,...), compute P(composite_token | factored_state)
        = P(token_i | component_state_1) * P(token_j | component_state_2) * ...
        """
        # Get probability distributions for each component
        component_probs = []
        for component, component_state in zip(self.components, state, strict=True):
            component_prob_dist = component.observation_probability_distribution(component_state)
            component_probs.append(component_prob_dist)

        # Compute outer product of all component probability distributions
        # For 2 components with vocab sizes V1, V2: result shape is (V1*V2,)
        composite_probs = jnp.ones(self.vocab_size)

        for composite_token in range(self.vocab_size):
            component_tokens = self._token_to_tuple(jnp.array(composite_token))
            prob = 1.0
            for component_token, component_prob_dist in zip(component_tokens, component_probs, strict=True):
                prob *= component_prob_dist[component_token]
            composite_probs = composite_probs.at[composite_token].set(prob)

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
        """Compute probability of generating observation sequence.

        Since components are independent, we can compute this by running each component
        independently and multiplying their probabilities.
        """
        # Extract component sequences from composite observations
        component_sequences = []
        for i in range(len(self.components)):
            component_seq = []
            for obs in observations:
                component_tokens = self._token_to_tuple(obs)
                component_seq.append(component_tokens[i])
            component_sequences.append(jnp.array(component_seq))

        # Compute probability for each component independently and multiply
        total_prob = jnp.array(1.0)
        for component, component_seq in zip(self.components, component_sequences, strict=True):
            component_prob = component.probability(component_seq)
            total_prob *= component_prob

        return total_prob

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute log probability of generating observation sequence."""
        # Extract component sequences
        component_sequences = []
        for i in range(len(self.components)):
            component_seq = []
            for obs in observations:
                component_tokens = self._token_to_tuple(obs)
                component_seq.append(component_tokens[i])
            component_sequences.append(jnp.array(component_seq))

        # Compute log probability for each component independently and sum
        total_log_prob = jnp.array(0.0)
        for component, component_seq in zip(self.components, component_sequences, strict=True):
            component_log_prob = component.log_probability(component_seq)
            total_log_prob += component_log_prob

        return total_log_prob
