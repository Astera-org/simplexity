import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.log_math import log_matmul, signed_logsumexp


def test_log_matmul():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    A = jax.random.uniform(key_a, (3, 4))
    B = jax.random.uniform(key_b, (4, 5))
    chex.assert_trees_all_close(log_matmul(jnp.log(A), jnp.log(B)), jnp.log(A @ B))


def test_signed_logsumexp():
    # values:
    # [ 4,  1,  2] ->  7
    # [-1, -3, -2] -> -6
    # [-1,  0,  2] ->  1

    log_abs_values = jnp.log(
        jnp.array(
            [
                [4, 1, 2],
                [1, 3, 2],
                [1, 0, 2],
            ]
        )
    )
    signs = jnp.array(
        [
            [1, 1, 1],
            [-1, -1, -1],
            [-1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    actual_log_abs_values, actual_signs = signed_logsumexp(log_abs_values, signs, axis=1)
    expected_log_abs_values = jnp.log(jnp.array([7, 6, 1]))
    expected_signs = jnp.array([1, -1, 1], dtype=jnp.int32)

    chex.assert_trees_all_close(actual_log_abs_values, expected_log_abs_values)
    chex.assert_trees_all_close(actual_signs, expected_signs)
