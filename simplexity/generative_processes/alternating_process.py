"""Alternating generative process that cycles through n different sub-processes.

This module provides a simple wrapper around HMM/GHMM that alternates deterministically
through n different transition matrix sets based on position.
"""

from __future__ import annotations

from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess

ComponentType = Literal["hmm", "ghmm"]

# State type: (position, belief_state)
AlternatingState = tuple[jnp.ndarray, jnp.ndarray]


class AlternatingProcess(GenerativeProcess[AlternatingState]):
    """Generative process that alternates through n different sub-processes.

    The process cycles deterministically through n different transition matrices:
    - With n_repetitions=1: position 0→variant 0, position 1→variant 1, etc.
    - With n_repetitions=k: positions 0..k-1→variant 0, positions k..2k-1→variant 1, etc.

    Each sub-process has its own transition matrix but shares the same state space
    and vocabulary size.

    State structure: (position_counter, belief_state)
    - position_counter: int tracking which step we're on
    - belief_state: probability distribution over hidden states [S]

    Attributes:
        component_type: Either "hmm" or "ghmm"
        transition_matrices: Array of shape [n, V, S, S] with n transition matrices
        normalizing_eigenvectors: Array of shape [n, S] for GHMM normalization
        initial_state_dist: Initial belief state distribution [S]
        n: Number of sub-processes to cycle through
        n_repetitions: Number of times each process is used before switching
        vocab_size: Vocabulary size (same across all variants)
        num_states: Number of hidden states (same across all variants)
    """

    component_type: ComponentType
    transition_matrices: jnp.ndarray  # [n, V, S, S]
    normalizing_eigenvectors: jnp.ndarray  # [n, S]
    initial_state_dist: jnp.ndarray  # [S]
    n: int
    n_repetitions: int

    def __init__(
        self,
        *,
        component_type: ComponentType,
        transition_matrices: jnp.ndarray,
        normalizing_eigenvectors: jnp.ndarray,
        initial_state: jnp.ndarray,
        n_repetitions: int = 1,
    ):
        """Initialize alternating process.

        Args:
            component_type: Either "hmm" or "ghmm"
            transition_matrices: Array of shape [n, V, S, S] with n variants
            normalizing_eigenvectors: Array of shape [n, S] for GHMM (ignored for HMM)
            initial_state: Initial belief state [S]
            n_repetitions: Number of times each variant is used consecutively
        """
        if transition_matrices.ndim != 4:
            raise ValueError(f"transition_matrices must have shape [n, V, S, S], got {transition_matrices.shape}")

        n, V, S1, S2 = transition_matrices.shape
        if S1 != S2:
            raise ValueError(f"State dimensions must match: {S1} vs {S2}")

        if initial_state.shape != (S1,):
            raise ValueError(f"initial_state shape {initial_state.shape} doesn't match state dim {S1}")

        if n_repetitions <= 0:
            raise ValueError(f"n_repetitions must be positive, got {n_repetitions}")

        self.component_type = component_type
        self.transition_matrices = transition_matrices
        self.normalizing_eigenvectors = normalizing_eigenvectors
        self.initial_state_dist = initial_state
        self.n = n
        self.n_repetitions = n_repetitions

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return int(self.transition_matrices.shape[1])

    @property
    def num_states(self) -> int:
        """Number of hidden states."""
        return int(self.transition_matrices.shape[2])

    @property
    def initial_state(self) -> AlternatingState:
        """Initial state: (position=0, initial_belief_state)."""
        return (jnp.array(0, dtype=jnp.int32), self.initial_state_dist)

    def _select_variant(self, position: jnp.ndarray) -> jnp.ndarray:
        """Select which variant to use based on position.

        Args:
            position: Current position counter

        Returns:
            Variant index k = (position // n_repetitions) % n (as JAX array)
        """
        return (position // self.n_repetitions) % self.n

    def _compute_obs_dist(self, belief_state: jnp.ndarray, variant_idx: jnp.ndarray) -> jnp.ndarray:
        """Compute observation distribution for given belief state and variant.

        Args:
            belief_state: Current belief state [S]
            variant_idx: Which variant to use (0 to n-1)

        Returns:
            Distribution over observations [V]
        """
        T = self.transition_matrices[variant_idx]  # [V, S, S]

        if self.component_type == "hmm":
            # HMM: P(obs|state) = sum_s belief[s] * sum_s' T[obs, s, s']
            obs_state = belief_state @ T  # [V, S]
            return jnp.sum(obs_state, axis=1)  # [V]
        else:  # ghmm
            # GHMM: weighted by normalizing eigenvector
            norm = self.normalizing_eigenvectors[variant_idx]  # [S]
            numer = belief_state @ T @ norm  # [V]
            denom = jnp.sum(belief_state * norm)  # scalar
            return numer / denom

    def _transition(self, belief_state: jnp.ndarray, obs: chex.Array, variant_idx: jnp.ndarray) -> jnp.ndarray:
        """Compute next belief state after observing a token.

        Args:
            belief_state: Current belief state [S]
            obs: Observed token (scalar)
            variant_idx: Which variant to use

        Returns:
            Updated belief state [S]
        """
        T = self.transition_matrices[variant_idx, obs]  # [S, S]

        if self.component_type == "hmm":
            # HMM: normalize by sum
            new_state = belief_state @ T  # [S]
            return new_state / jnp.sum(new_state)
        else:  # ghmm
            # GHMM: normalize by eigenvector
            norm = self.normalizing_eigenvectors[variant_idx]  # [S]
            new_state = belief_state @ T  # [S]
            return (new_state * norm) / jnp.sum(new_state * norm)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: AlternatingState) -> jnp.ndarray:
        """Compute P(obs | state).

        Args:
            state: (position, belief_state)

        Returns:
            Distribution over observations [V]
        """
        position, belief_state = state
        variant_idx = self._select_variant(position)
        return self._compute_obs_dist(belief_state, variant_idx)

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: AlternatingState) -> jnp.ndarray:
        """Compute log P(obs | state).

        Args:
            log_belief_state: (position, log_belief_state)

        Returns:
            Log distribution over observations [V]
        """
        position, log_belief = log_belief_state
        belief_state = jnp.exp(log_belief)
        variant_idx = self._select_variant(position)
        probs = self._compute_obs_dist(belief_state, variant_idx)
        return jnp.log(probs)

    @eqx.filter_jit
    def emit_observation(self, state: AlternatingState, key: jax.Array) -> jnp.ndarray:
        """Sample observation from current state.

        Args:
            state: (position, belief_state)
            key: JAX random key

        Returns:
            Sampled observation (scalar token)
        """
        probs = self.observation_probability_distribution(state)
        return jax.random.categorical(key, jnp.log(probs))

    @eqx.filter_jit
    def transition_states(self, state: AlternatingState, obs: chex.Array) -> AlternatingState:
        """Update state after observing a token.

        Args:
            state: Current (position, belief_state)
            obs: Observed token

        Returns:
            Updated (position+1, new_belief_state)
        """
        position, belief_state = state
        variant_idx = self._select_variant(position)
        new_belief_state = self._transition(belief_state, obs, variant_idx)
        new_position = position + 1
        return (new_position, new_belief_state)

    @eqx.filter_jit
    def probability(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Compute P(observations).

        Args:
            observations: Array of observations

        Returns:
            Scalar probability
        """

        def step(carry: AlternatingState, obs: jnp.ndarray):
            state = carry
            dist = self.observation_probability_distribution(state)
            p = dist[obs]
            new_state = self.transition_states(state, obs)
            return new_state, p

        _, ps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.prod(ps)

    @eqx.filter_jit
    def log_probability(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Compute log P(observations).

        Args:
            observations: Array of observations

        Returns:
            Scalar log probability
        """

        def step(carry: AlternatingState, obs: jnp.ndarray):
            position, belief_state = carry
            log_belief_state = jnp.log(belief_state)
            log_state = (position, log_belief_state)
            log_dist = self.log_observation_probability_distribution(log_state)
            lp = log_dist[obs]
            new_state = self.transition_states(carry, obs)
            return new_state, lp

        _, lps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.sum(lps)
