import chex
import jax
import jax.numpy as jnp


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
