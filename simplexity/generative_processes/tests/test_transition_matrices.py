import jax.numpy as jnp

from simplexity.generative_processes.transition_matrices import (
    days_of_week,
    fanizza,
    mess3,
    post_quantum,
    rrxor,
    tom_quantum,
)


def test_post_quantum():
    transition_matrices = post_quantum()
    assert transition_matrices.shape == (3, 3, 3)
    # Verify that transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum_normalized = transition_matrices.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrix_sum_normalized)).max()
    assert jnp.isclose(transition_matrix_sum_max_eigval, 1, atol=1e-10), "Largest absolute eigenvalue is not 1"


def test_days_of_week():
    transition_matrices = days_of_week()
    assert transition_matrices.shape == (11, 7, 7)


def test_tom_quantum():
    transition_matrices = tom_quantum(alpha=1.0, beta=1.0)
    assert transition_matrices.shape == (4, 3, 3)


def test_fanizza():
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    assert transition_matrices.shape == (2, 4, 4)
    tau = jnp.ones(4)
    assert jnp.allclose(transition_matrices[0] @ tau + transition_matrices[1] @ tau, tau), (
        "Stochasticity condition not met"
    )


def test_rrxor():
    transition_matrices = rrxor()
    assert transition_matrices.shape == (2, 5, 5)


def test_mess3():
    transition_matrices = mess3()
    assert transition_matrices.shape == (3, 3, 3)
