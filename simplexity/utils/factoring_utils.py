"""Core computational kernels for HMM/GHMM factor operations.

These functions implement the observation and transition dynamics
for individual factors, supporting both HMM and GHMM variants.
"""

from __future__ import annotations

from typing import Literal

import chex
import jax.numpy as jnp
import equinox as eqx
import jax.numpy as jnp

ComponentType = Literal["hmm", "ghmm"]


def compute_obs_dist_for_variant(
    component_type: ComponentType,
    state: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    normalizing_eigenvector: jnp.ndarray | None = None,
) -> jnp.ndarray:
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
        return jnp.sum(obs_state, axis=1)  # [V]
    else:  # ghmm
        # GHMM: normalize by eigenvector
        if normalizing_eigenvector is None:
            raise ValueError("GHMM requires normalizing_eigenvector")
        numer = state @ transition_matrix @ normalizing_eigenvector  # [V]
        denom = jnp.sum(state * normalizing_eigenvector)  # scalar
        return numer / denom


def transition_with_obs(
    component_type: ComponentType,
    state: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    obs: jnp.ndarray,
    normalizing_eigenvector: jnp.ndarray | None = None,
) -> jnp.ndarray:
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

"""Token encoding utilities for factored observations.

Handles conversion between composite tokens and per-factor token tuples
using radix/base conversion.
"""


class TokenEncoder(eqx.Module):
    """Encodes/decodes composite observations from per-factor tokens.

    Uses radix encoding: given vocab sizes [V_0, V_1, ..., V_{F-1}],
    a tuple (t_0, t_1, ..., t_{F-1}) maps to:
        composite = t_0 * (V_1 * V_2 * ... * V_{F-1}) + t_1 * (V_2 * ... * V_{F-1}) + ... + t_{F-1}

    Attributes:
        vocab_sizes: Array of shape [F] with vocabulary size per factor
        radix_multipliers: Array of shape [F] with multipliers for encoding
    """

    vocab_sizes: jnp.ndarray  # shape [F]
    radix_multipliers: jnp.ndarray  # shape [F]

    def __init__(self, vocab_sizes: jnp.ndarray):
        """Initialize encoder with vocab sizes.

        Args:
            vocab_sizes: Array of shape [F] with vocabulary size per factor
        """
        self.vocab_sizes = jnp.asarray(vocab_sizes)

        # Compute radix multipliers
        F = len(vocab_sizes)
        multipliers = []
        for i in range(F):
            m = 1
            for j in range(i + 1, F):
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

    def tuple_to_token(self, token_tuple: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Convert per-factor tokens to composite token.

        Args:
            token_tuple: Tuple of F scalar arrays, each in [0, V_i)

        Returns:
            Scalar array with composite token in [0, prod(V_i))
        """
        token = jnp.array(0)
        multiplier = jnp.array(1)
        for i in reversed(range(len(token_tuple))):
            token += token_tuple[i] * multiplier
            multiplier *= self.vocab_sizes[i]
        return token

    def token_to_tuple(self, token: chex.Array) -> tuple[jnp.ndarray, ...]:
        """Convert composite token to per-factor tokens.

        Args:
            token: Scalar array with composite token

        Returns:
            Tuple of F scalar arrays with per-factor tokens
        """
        result = []
        remaining = jnp.array(token)
        for i in reversed(range(self.num_factors)):
            v = self.vocab_sizes[i]
            t_i = remaining % v
            result.append(t_i)
            remaining = remaining // v
        return tuple(reversed(result))

    def extract_factors_vectorized(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Extract per-factor tokens from batch of composite tokens.

        Args:
            tokens: Array of shape [N] with composite tokens

        Returns:
            Array of shape [N, F] with per-factor tokens
        """
        tokens = jnp.atleast_1d(tokens)
        return (tokens[:, None] // self.radix_multipliers[None, :]) % self.vocab_sizes[None, :]
