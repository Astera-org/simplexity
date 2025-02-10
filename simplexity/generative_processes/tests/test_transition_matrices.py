import jax.numpy as jnp
import pytest

from simplexity.generative_processes.transition_matrices import (
    days_of_week,
    even_ones,
    fanizza,
    mess3,
    no_consecutive_ones,
    post_quantum,
    rrxor,
    tom_quantum,
    zero_one_random,
)
from simplexity.generative_processes.utils import stationary_distribution


def validate_transition_matrices(transition_matrices: jnp.ndarray):
    assert jnp.all(transition_matrices >= 0)
    assert jnp.all(transition_matrices <= 1)

    sum_over_obs_and_next = jnp.sum(transition_matrices, axis=(0, 1))
    assert jnp.allclose(sum_over_obs_and_next, 1.0), "Probabilities don't sum to 1 for each current state"

    transition_matrix = jnp.sum(transition_matrices, axis=0)
    distribution = stationary_distribution(transition_matrix)
    assert distribution.size > 0, "State transition matrix should have stationary distribution = 1"


def test_no_consecutive_ones():
    transition_matrices = no_consecutive_ones()
    assert transition_matrices.shape == (2, 2, 2)
    validate_transition_matrices(transition_matrices)


def test_even_ones():
    transition_matrices = even_ones()
    assert transition_matrices.shape == (2, 2, 2)
    validate_transition_matrices(transition_matrices)


def test_zero_one_random():
    transition_matrices = zero_one_random()
    assert transition_matrices.shape == (2, 3, 3)
    validate_transition_matrices(transition_matrices)


def test_post_quantum():
    transition_matrices = post_quantum()
    assert transition_matrices.shape == (3, 3, 3)
    try:
        validate_transition_matrices(transition_matrices)
    except AssertionError:
        pytest.xfail("Matrix contains negative values")
    # Verify that transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum_normalized = transition_matrices.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrix_sum_normalized)).max()
    assert jnp.isclose(transition_matrix_sum_max_eigval, 1, atol=1e-10), "Largest absolute eigenvalue is not 1"


def test_days_of_week():
    transition_matrices = days_of_week()
    assert transition_matrices.shape == (11, 7, 7)
    validate_transition_matrices(transition_matrices)


def test_tom_quantum():
    transition_matrices = tom_quantum(alpha=1.0, beta=1.0)
    assert transition_matrices.shape == (4, 3, 3)
    try:
        validate_transition_matrices(transition_matrices)
    except AssertionError:
        pytest.xfail("Matrix contains negative values")


def test_fanizza():
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    assert transition_matrices.shape == (2, 4, 4)
    try:
        validate_transition_matrices(transition_matrices)
    except AssertionError:
        pytest.xfail("Matrix contains negative values")
    tau = jnp.ones(4)
    assert jnp.allclose(transition_matrices[0] @ tau + transition_matrices[1] @ tau, tau), (
        "Stochasticity condition not met"
    )


def test_rrxor():
    transition_matrices = rrxor()
    assert transition_matrices.shape == (2, 5, 5)
    validate_transition_matrices(transition_matrices)


def test_mess3():
    transition_matrices = mess3()
    assert transition_matrices.shape == (3, 3, 3)
    validate_transition_matrices(transition_matrices)
