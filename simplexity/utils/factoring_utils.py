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
    """Compute observation distribution for a single factor variant."""
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
    """Update state after observing a token."""
    new_state = state @ transition_matrix[obs]  # [S]

    if component_type == "hmm":
        # HMM: normalize by sum
        return new_state / jnp.sum(new_state)
    else:  # ghmm
        # GHMM: normalize by eigenvector
        if normalizing_eigenvector is None:
            raise ValueError("GHMM requires normalizing_eigenvector")
        return new_state / (new_state @ normalizing_eigenvector)


def _radix_multipliers(vs: jax.Array) -> jax.Array:
    """Compute radix multipliers for an array of vocab sizes."""
    suffixes = jnp.cumprod(vs[::-1])[::-1]
    return suffixes // vs


def compute_other_multipliers(vocab_sizes: tuple[int, ...]) -> tuple[jax.Array, ...]:
    """Compute radix multipliers for other-factor indexing."""
    vs = jnp.array(vocab_sizes)
    num_factors = len(vocab_sizes)
    result = []
    for i in range(num_factors):
        other_vs = jnp.concatenate([vs[:i], vs[i + 1 :]])
        if len(other_vs) == 0:
            result.append(jnp.zeros(num_factors, dtype=jnp.int32))
        else:
            mults = _radix_multipliers(other_vs)
            result.append(jnp.concatenate([mults[:i], jnp.array([0]), mults[i:]]))
    return tuple(result)


def compute_prefix_multipliers(vocab_sizes: tuple[int, ...]) -> tuple[jax.Array, ...]:
    """Compute radix multipliers for prefix-factor indexing."""
    vs = jnp.array(vocab_sizes)
    num_factors = len(vocab_sizes)
    result = []
    for i in range(num_factors):
        if i == 0:
            result.append(jnp.zeros(num_factors, dtype=jnp.int32))
        else:
            mults = _radix_multipliers(vs[:i])
            result.append(jnp.concatenate([mults, jnp.zeros(num_factors - i, dtype=jnp.int32)]))
    return tuple(result)


class TokenEncoder(eqx.Module):
    """Encodes/decodes composite observations from per-factor tokens using radix encoding."""

    vocab_sizes: jax.Array  # shape [F]
    radix_multipliers: jax.Array  # shape [F]

    def __init__(self, vocab_sizes: jax.Array):
        self.vocab_sizes = jnp.asarray(vocab_sizes)
        self.radix_multipliers = _radix_multipliers(self.vocab_sizes)

    @property
    def num_factors(self) -> int:
        """Number of factors."""
        return int(self.vocab_sizes.shape[0])

    @property
    def composite_vocab_size(self) -> int:
        """Total vocabulary size of composite observation."""
        return int(jnp.prod(self.vocab_sizes))

    def tuple_to_token(self, token_tuple: tuple[jax.Array, ...]) -> jax.Array:
        """Convert per-factor tokens to a composite token."""
        token = jnp.array(0)
        multiplier = jnp.array(1)
        for i in reversed(range(len(token_tuple))):
            token += token_tuple[i] * multiplier
            multiplier *= self.vocab_sizes[i]
        return token

    def token_to_tuple(self, token: chex.Array) -> tuple[jax.Array, ...]:
        """Convert a composite token to per-factor tokens."""
        result = []
        remaining = jnp.array(token)
        for i in reversed(range(self.num_factors)):
            v = self.vocab_sizes[i]
            t_i = remaining % v
            result.append(t_i)
            remaining = remaining // v
        return tuple(reversed(result))

    def extract_factors_vectorized(self, tokens: jax.Array) -> jax.Array:
        """Extract per-factor tokens from a batch of composite tokens."""
        tokens = jnp.atleast_1d(tokens)
        return (tokens[:, None] // self.radix_multipliers[None, :]) % self.vocab_sizes[None, :]
