import chex
import jax.numpy as jnp

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
from tests.assertions import assert_proportional


def validate_ghmm_transition_matrices(
    transition_matrices: jnp.ndarray, rtol: float = 1e-6, atol: float = 0
):
    transition_matrix = jnp.sum(transition_matrices, axis=0)
    num_states = transition_matrix.shape[0]

    eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)
    assert jnp.isclose(jnp.max(eigenvalues), 1.0), (
        "State transition matrix should have eigenvalue = 1"
    )
    normalizing_eigenvector = (
        right_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze().real
    )
    assert normalizing_eigenvector.shape == (num_states,)

    eigenvalues, left_eigenvectors = jnp.linalg.eig(transition_matrix.T)
    assert jnp.isclose(jnp.max(eigenvalues), 1.0), (
        "State transition matrix should have eigenvalue = 1"
    )
    stationary_state = left_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze().real
    assert stationary_state.shape == (num_states,)


def validate_hmm_transition_matrices(
    transition_matrices: jnp.ndarray, rtol: float = 1e-6, atol: float = 0
):
    validate_ghmm_transition_matrices(transition_matrices, rtol, atol)
    assert jnp.all(transition_matrices >= 0)
    assert jnp.all(transition_matrices <= 1)

    sum_over_obs_and_next = jnp.sum(transition_matrices, axis=(0, 2))
    chex.assert_trees_all_close(
        sum_over_obs_and_next,
        jnp.ones_like(sum_over_obs_and_next),
        rtol=rtol,
        atol=atol,
    )

    transition_matrix = jnp.sum(transition_matrices, axis=0)
    eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)
    normalizing_eigenvector = (
        right_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze().real
    )
    assert_proportional(
        normalizing_eigenvector,
        jnp.ones_like(normalizing_eigenvector),
        rtol=rtol,
        atol=atol,
    )


def test_no_consecutive_ones():
    transition_matrices = no_consecutive_ones()
    assert transition_matrices.shape == (2, 2, 2)
    validate_hmm_transition_matrices(transition_matrices)


def test_even_ones():
    transition_matrices = even_ones()
    assert transition_matrices.shape == (2, 2, 2)
    validate_hmm_transition_matrices(transition_matrices)


def test_zero_one_random():
    transition_matrices = zero_one_random()
    assert transition_matrices.shape == (2, 3, 3)
    validate_hmm_transition_matrices(transition_matrices)


def test_post_quantum():
    transition_matrices = post_quantum()
    assert transition_matrices.shape == (3, 3, 3)
    validate_ghmm_transition_matrices(transition_matrices)
    # Verify that transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum_normalized = transition_matrices.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(
        jnp.linalg.eigvals(transition_matrix_sum_normalized)
    ).max()
    assert jnp.isclose(transition_matrix_sum_max_eigval, 1, atol=1e-10), (
        "Largest absolute eigenvalue is not 1"
    )


def test_days_of_week():
    transition_matrices = days_of_week()
    assert transition_matrices.shape == (11, 7, 7)
    validate_hmm_transition_matrices(transition_matrices, rtol=2e-6)


def test_tom_quantum():
    transition_matrices = tom_quantum(alpha=1.0, beta=1.0)
    assert transition_matrices.shape == (4, 3, 3)
    validate_ghmm_transition_matrices(transition_matrices)


def test_fanizza():
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    assert transition_matrices.shape == (2, 4, 4)
    validate_ghmm_transition_matrices(transition_matrices)
    tau = jnp.ones(4)
    assert jnp.allclose(jnp.sum(transition_matrices @ tau, axis=0), tau), (
        "Stochasticity condition not met"
    )


def test_rrxor():
    transition_matrices = rrxor()
    assert transition_matrices.shape == (2, 5, 5)
    validate_hmm_transition_matrices(transition_matrices)


def test_mess3():
    transition_matrices = mess3()
    assert transition_matrices.shape == (3, 3, 3)
    validate_hmm_transition_matrices(transition_matrices)
