"""Core computational kernels for HMM/GHMM factor operations.

These functions implement the observation and transition dynamics
for individual factors, supporting both HMM and GHMM variants.
"""

from __future__ import annotations

from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

ComponentType = Literal["hmm", "ghmm"]


def compute_obs_dist_for_variant(
    component_type: ComponentType,
    state: jax.Array,
    transition_matrix: jax.Array,
    normalizing_eigenvector: jax.Array | None = None,
) -> jax.Array:
    """Compute observation distribution for a single factor variant.

    Args:
        component_type: "hmm" or "ghmm"
        state: State vector of shape [S]
        transition_matrix: Transition tensor of shape [V, S, S]
        normalizing_eigenvector: For GHMM only, shape [S]. Ignored for HMM.

    Returns:
        Distribution over observations, shape [V]
    """
    if component_type == "hmm":
        # HMM: normalize by sum
        obs_state = state @ transition_matrix  # [V, S]
        probs = jnp.sum(obs_state, axis=1)  # [V]
    else:  # ghmm
        # GHMM: normalize by eigenvector
        if normalizing_eigenvector is None:
            raise ValueError("GHMM requires normalizing_eigenvector")
        numer = state @ transition_matrix @ normalizing_eigenvector  # [V]
        denom = jnp.sum(state * normalizing_eigenvector)  # scalar
        probs = numer / denom

    # Clamp to non-negative to handle numerical precision issues
    # (small negative values can arise from GHMM eigenvector computations)
    return jnp.maximum(probs, 0.0)


def transition_with_obs(
    component_type: ComponentType,
    state: jax.Array,
    transition_matrix: jax.Array,
    obs: jax.Array,
    normalizing_eigenvector: jax.Array | None = None,
) -> jax.Array:
    """Update state after observing a token.

    Args:
        component_type: "hmm" or "ghmm"
        state: Current state vector of shape [S]
        transition_matrix: Transition tensor of shape [V, S, S]
        obs: Observed token (scalar int)
        normalizing_eigenvector: For GHMM only, shape [S]. Ignored for HMM.

    Returns:
        New normalized state vector of shape [S]
    """
    new_state = state @ transition_matrix[obs]  # [S]

    if component_type == "hmm":
        # HMM: normalize by sum
        return new_state / jnp.sum(new_state)
    else:  # ghmm
        # GHMM: normalize by eigenvector
        if normalizing_eigenvector is None:
            raise ValueError("GHMM requires normalizing_eigenvector")
        return new_state / (new_state @ normalizing_eigenvector)


class TokenEncoder(eqx.Module):
    """Encodes/decodes composite observations from per-factor tokens.

    Uses radix encoding: given vocab sizes [V_0, V_1, ..., V_{F-1}],
    a tuple (t_0, t_1, ..., t_{F-1}) maps to:
        composite = t_0 * (V_1 * V_2 * ... * V_{F-1}) + t_1 * (V_2 * ... * V_{F-1}) + ... + t_{F-1}

    Attributes:
        vocab_sizes: Array of shape [F] with vocabulary size per factor
        radix_multipliers: Array of shape [F] with multipliers for encoding
    """

    vocab_sizes: jax.Array  # shape [F]
    radix_multipliers: jax.Array  # shape [F]

    def __init__(self, vocab_sizes: jax.Array):
        """Initialize encoder with vocab sizes.

        Args:
            vocab_sizes: Array of shape [F] with vocabulary size per factor
        """
        self.vocab_sizes = jnp.asarray(vocab_sizes)

        # Compute radix multipliers
        f = len(vocab_sizes)
        multipliers = []
        for i in range(f):
            m = 1
            for j in range(i + 1, f):
                m *= int(vocab_sizes[j])
            multipliers.append(m)
        self.radix_multipliers = jnp.array(multipliers)

    @property
    def num_factors(self) -> int:
        """Number of factors."""
        return int(self.vocab_sizes.shape[0])

    @property
    def composite_vocab_size(self) -> int:
        """Total vocabulary size of composite observation."""
        return int(jnp.prod(self.vocab_sizes))

    def tuple_to_token(self, token_tuple: tuple[jax.Array, ...]) -> jax.Array:
        """Convert per-factor tokens to composite token.

        Args:
            token_tuple: Tuple of f scalar arrays, each in [0, V_i)

        Returns:
            Scalar array with composite token in [0, prod(V_i))
        """
        token = jnp.array(0)
        multiplier = jnp.array(1)
        for i in reversed(range(len(token_tuple))):
            token += token_tuple[i] * multiplier
            multiplier *= self.vocab_sizes[i]
        return token

    def token_to_tuple(self, token: chex.Array) -> tuple[jax.Array, ...]:
        """Convert composite token to per-factor tokens.

        Args:
            token: Scalar array with composite token

        Returns:
            Tuple of f scalar arrays with per-factor tokens
        """
        result = []
        remaining = jnp.array(token)
        for i in reversed(range(self.num_factors)):
            v = self.vocab_sizes[i]
            t_i = remaining % v
            result.append(t_i)
            remaining = remaining // v
        return tuple(reversed(result))

    def extract_factors_vectorized(self, tokens: jax.Array) -> jax.Array:
        """Extract per-factor tokens from batch of composite tokens.

        Args:
            tokens: Array of shape [n] with composite tokens

        Returns:
            Array of shape [n, f] with per-factor tokens
        """
        tokens = jnp.atleast_1d(tokens)
        return (tokens[:, None] // self.radix_multipliers[None, :]) % self.vocab_sizes[None, :]
