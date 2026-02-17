"""Inflated vocabulary generative process wrapper.

Wraps any GenerativeProcess by adding a uniform noise dimension to the vocabulary,
increasing vocab size by a multiplicative factor K. The noise dimension is stateless:
state dynamics depend only on the base token.

Token encoding: inflated_token = noise_prefix * V_base + base_token
- base_token extraction: inflated_token % V_base
- noise_prefix extraction: inflated_token // V_base
"""

from __future__ import annotations

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


class InflatedVocabularyProcess[State](GenerativeProcess[State]):
    """Wraps a GenerativeProcess by adding a uniform noise dimension to inflate the vocabulary.

    For a base process with vocab size V and inflation factor K:
    - New vocab size is K * V
    - inflated_token = noise_prefix * V + base_token
    - P(inflated_token | state) = P(base_token | state) / K
    - State dynamics only depend on base_token (noise is stateless)

    This increases optimal per-token loss by exactly log(K) nats.

    Args:
        base_process: The generative process to wrap.
        inflation_factor: Number of noise variants per base token (K >= 2).
    """

    base_process: GenerativeProcess[State]
    inflation_factor: int
    _base_vocab_size: int
    _inflated_vocab_size: int

    def __init__(
        self,
        base_process: GenerativeProcess[State],
        inflation_factor: int,
    ) -> None:
        if inflation_factor < 2:
            raise ValueError(f"inflation_factor must be >= 2, got {inflation_factor}")
        self.base_process = base_process
        self.inflation_factor = inflation_factor
        self._base_vocab_size = base_process.vocab_size
        self._inflated_vocab_size = inflation_factor * base_process.vocab_size

    @property
    def vocab_size(self) -> int:
        """The number of inflated observations: K * base vocab size."""
        return self._inflated_vocab_size

    @property
    def initial_state(self) -> State:
        """The initial state, identical to the base process."""
        return self.base_process.initial_state

    @eqx.filter_jit
    def emit_observation(self, state: State, key: chex.PRNGKey) -> chex.Array:
        """Emit an inflated observation: sample base token then add uniform noise prefix."""
        k1, k2 = jax.random.split(key)
        base_obs = self.base_process.emit_observation(state, k1)
        noise_prefix = jax.random.randint(k2, (), 0, self.inflation_factor)
        return noise_prefix * self._base_vocab_size + base_obs

    @eqx.filter_jit
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Update state using only the base token (noise prefix is discarded)."""
        base_obs = jnp.mod(obs, self._base_vocab_size)
        return self.base_process.transition_states(state, base_obs)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute P(inflated_obs | state) = P(base_obs | state) / K for each noise variant."""
        base_dist = self.base_process.observation_probability_distribution(state)
        return jnp.tile(base_dist / self.inflation_factor, self.inflation_factor)

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: State) -> jax.Array:
        """Compute log P(inflated_obs | state) = log P(base_obs | state) - log(K)."""
        base_log_dist = self.base_process.log_observation_probability_distribution(log_belief_state)
        return jnp.tile(base_log_dist - jnp.log(self.inflation_factor), self.inflation_factor)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute P(inflated_seq) = P(base_seq) * (1/K)^T."""
        base_obs = jnp.mod(observations, self._base_vocab_size)
        base_prob = self.base_process.probability(base_obs)
        return base_prob * (1.0 / self.inflation_factor) ** observations.shape[0]

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute log P(inflated_seq) = log P(base_seq) - T * log(K)."""
        base_obs = jnp.mod(observations, self._base_vocab_size)
        base_log_prob = self.base_process.log_probability(base_obs)
        return base_log_prob - observations.shape[0] * jnp.log(self.inflation_factor)
