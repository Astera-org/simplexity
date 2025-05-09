from collections.abc import Iterable, Mapping
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

Tree = jax.Array | jnp.ndarray | bool | jnp.number | Iterable[chex.ArrayTree] | Mapping[Any, chex.ArrayTree]


def assert_proportional(a: jax.Array, b: jax.Array, rtol: float = 1e-6, atol: float = 0):
    """Assert that two arrays are proportional."""

    def normalize(x: jax.Array) -> jax.Array:
        return x / jnp.maximum(jnp.abs(x).max(), 1e-6)

    chex.assert_equal_shape([a, b])
    norm_a = normalize(a)
    norm_b = normalize(b)
    try:
        chex.assert_trees_all_close(norm_a, norm_b, rtol=rtol, atol=atol)
    except AssertionError as e1:
        try:
            chex.assert_trees_all_close(norm_a, -norm_b, rtol=rtol, atol=atol)
        except AssertionError as e2:
            if jnp.sum(jnp.abs(norm_b - norm_a)) > jnp.sum(jnp.abs(norm_b + norm_a)):
                e = e2
            else:
                e = e1
            raise AssertionError(f"Arrays are not proportional: {a} and {b}.\n{e}") from e


def assert_trees_different(a: Tree, b: Tree, rtol: float = 1e-6, atol: float = 0):
    """Assert that two arrays are different."""
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(a, b, rtol=rtol, atol=atol)
