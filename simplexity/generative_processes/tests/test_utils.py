import jax
import jax.numpy as jnp

from simplexity.generative_processes.utils import log_matmul, normalize_simplex, stationary_distribution


def test_log_matmul():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    A = jax.random.uniform(key_a, (3, 4))
    B = jax.random.uniform(key_b, (4, 5))
    assert jnp.allclose(log_matmul(jnp.log(A), jnp.log(B)), jnp.log(A @ B))


def test_normalize_simplex():
    simplex = jnp.array([1, 3, 4])
    expected = jnp.array([0.125, 0.375, 0.5])
    assert jnp.allclose(normalize_simplex(simplex), expected)
    assert jnp.allclose(normalize_simplex(-simplex), expected)


def test_stationary_distribution():
    transition_matrix = jnp.array([
        [0, 0, 0, 1], 
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    expected = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(stationary_distribution(transition_matrix), expected)
