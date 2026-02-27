"""Shared indexing helpers for conditional factor structures."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def build_other_factor_multipliers(vocab_sizes: tuple[int, ...]) -> tuple[jax.Array, ...]:
    """Build radix multipliers for flattening all tokens except one factor.

    Args:
        vocab_sizes: Per-factor vocabulary sizes.

    Returns:
        Tuple of length ``F`` where element ``i`` is an array of shape ``[F]``.
        Entry ``j`` is the radix multiplier for factor ``j`` in a flattened
        index over all factors except ``i``. Entry ``i`` is zero (unused).
    """
    num_factors = len(vocab_sizes)
    other_multipliers: list[jax.Array] = []
    for i in range(num_factors):
        mult = []
        for j in range(num_factors):
            if j == i:
                mult.append(0)
                continue

            m = 1
            for k in range(j + 1, num_factors):
                if k == i:
                    continue
                m *= vocab_sizes[k]
            mult.append(m)
        other_multipliers.append(jnp.array(mult, dtype=jnp.int32))
    return tuple(other_multipliers)


def flatten_index(tokens: jax.Array, multipliers: jax.Array) -> jax.Array:
    """Flatten token tuple to a scalar index with precomputed multipliers."""
    return jnp.sum(tokens * multipliers)
