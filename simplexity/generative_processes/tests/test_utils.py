import jax.numpy as jnp

from simplexity.generative_processes.utils import normalize_simplex, stationary_distribution


def test_normalize_simplex():
    simplex = jnp.array([1, 3, 4])
    expected = jnp.array([0.125, 0.375, 0.5])
    assert jnp.allclose(normalize_simplex(simplex), expected)
    assert jnp.allclose(normalize_simplex(-simplex), expected)


def test_stationary_distribution():
    transition_matrix = jnp.array(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    expected = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(stationary_distribution(transition_matrix), expected)
