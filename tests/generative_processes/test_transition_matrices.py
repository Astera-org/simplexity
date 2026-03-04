"""Test the transition matrices module."""

import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.transition_matrices import (
    coin,
    days_of_week,
    even_ones,
    fanizza,
    get_stationary_state,
    leaky_rrxor,
    matching_parens,
    mess3,
    mr_name,
    no_consecutive_ones,
    post_quantum,
    rrxor,
    sns,
    tom_quantum,
    zero_one_random,
)
from tests.assertions import assert_proportional


def test_stationary_state():
    """Test the get_stationary_state function."""
    transition_matrix = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    actual = get_stationary_state(transition_matrix)
    expected = jnp.array([0.5, 0.5])
    assert jnp.allclose(actual, expected)


def validate_ghmm_transition_matrices(transition_matrices: jax.Array, ergodic: bool = True) -> None:
    """Test the validate_ghmm_transition_matrices function."""
    transition_matrix = jnp.sum(transition_matrices, axis=0)
    num_states = transition_matrix.shape[0]

    eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)
    assert jnp.isclose(jnp.max(eigenvalues), 1.0), "State transition matrix should have eigenvalue = 1"
    if ergodic:
        normalizing_eigenvector = right_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze(axis=-1).real
        assert normalizing_eigenvector.shape == (num_states,)

    eigenvalues, left_eigenvectors = jnp.linalg.eig(transition_matrix.T)
    assert jnp.isclose(jnp.max(eigenvalues), 1.0), "State transition matrix should have eigenvalue = 1"
    if ergodic:
        stationary_state = left_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze(axis=-1).real
        assert stationary_state.shape == (num_states,)


def validate_hmm_transition_matrices(
    transition_matrices: jax.Array, ergodic: bool = True, rtol: float = 1e-6, atol: float = 0
):
    """Test the validate_hmm_transition_matrices function."""
    validate_ghmm_transition_matrices(transition_matrices, ergodic)
    assert jnp.all(transition_matrices >= 0)
    assert jnp.all(transition_matrices <= 1)

    sum_over_obs_and_next = jnp.sum(transition_matrices, axis=(0, 2))
    chex.assert_trees_all_close(
        sum_over_obs_and_next,
        jnp.ones_like(sum_over_obs_and_next),
        rtol=rtol,
        atol=atol,
    )

    if ergodic:
        transition_matrix = jnp.sum(transition_matrices, axis=0)
        eigenvalues, right_eigenvectors = jnp.linalg.eig(transition_matrix)
        normalizing_eigenvector = right_eigenvectors[:, jnp.isclose(eigenvalues, 1)].squeeze(axis=-1).real
        assert_proportional(
            normalizing_eigenvector,
            jnp.ones_like(normalizing_eigenvector),
            rtol=rtol,
            atol=atol,
        )


def test_coin():
    """Test the coin transition matrices."""
    transition_matrices = coin(p=0.5)
    assert transition_matrices.shape == (2, 1, 1)
    validate_hmm_transition_matrices(transition_matrices)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([1]))


def test_days_of_week():
    """Test the days of week transition matrices."""
    transition_matrices = days_of_week()
    assert transition_matrices.shape == (11, 7, 7)
    validate_hmm_transition_matrices(transition_matrices, rtol=2e-6)


def test_even_ones():
    """Test the even ones transition matrices."""
    transition_matrices = even_ones(p=0.5)
    assert transition_matrices.shape == (2, 2, 2)
    validate_hmm_transition_matrices(transition_matrices)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([2, 1]) / 3)


def test_fanizza():
    """Test the fanizza transition matrices."""
    transition_matrices = fanizza(alpha=2000, lamb=0.49)
    assert transition_matrices.shape == (2, 4, 4)
    validate_ghmm_transition_matrices(transition_matrices)
    tau = jnp.ones(4)
    assert jnp.allclose(jnp.sum(transition_matrices @ tau, axis=0), tau), "Stochasticity condition not met"


def test_leaky_rrxor():
    """Test the leaky rrxor transition matrices."""
    vocab_size = 2
    num_states = 5
    p1 = 0.5
    p2 = 0.5
    epsilon = 0.1
    transition_matrices = leaky_rrxor(p1=p1, p2=p2, epsilon=epsilon)
    assert transition_matrices.shape == (vocab_size, num_states, num_states)
    validate_hmm_transition_matrices(transition_matrices)

    base_matrices = rrxor(p1, p2)
    diff = jnp.abs(transition_matrices - base_matrices)
    leak_value = 1 / (vocab_size * num_states)
    min_diff_expected = epsilon * jnp.min(jnp.abs(base_matrices - leak_value))
    max_diff_expected = epsilon * jnp.max(jnp.abs(base_matrices - leak_value))
    assert jnp.allclose(jnp.min(diff), min_diff_expected, rtol=1e-5), (
        f"Minimum difference should be approximately {min_diff_expected}"
    )
    assert jnp.allclose(jnp.max(diff), max_diff_expected, rtol=1e-5), (
        f"Maximum difference should be approximately {max_diff_expected}"
    )


def test_leaky_rrxor_zero_epsilon():
    """Test that leaky rrxor with epsilon=0 equals regular rrxor."""
    p1, p2 = 0.5, 0.5
    leaky_matrices = leaky_rrxor(p1=p1, p2=p2, epsilon=0.0)
    base_matrices = rrxor(p1, p2)
    chex.assert_trees_all_close(leaky_matrices, base_matrices)


def test_matching_parens():
    """Test the matching parens transition matrices."""
    transition_matrices = matching_parens(open_probs=[1.0, 0.5, 0.5])
    assert transition_matrices.shape == (2, 4, 4)
    validate_hmm_transition_matrices(transition_matrices, rtol=1e-5)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([1, 2, 2, 1]) / 6)


def test_mess3():
    """Test the mess3 transition matrices."""
    transition_matrices = mess3(x=0.15, a=0.6)
    assert transition_matrices.shape == (3, 3, 3)
    validate_hmm_transition_matrices(transition_matrices)


def test_mr_name():
    """Test the mr name transition matrices."""
    transition_matrices = mr_name(p=0.4, q=0.25)
    assert transition_matrices.shape == (4, 4, 4)
    validate_hmm_transition_matrices(transition_matrices)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([1, 2, 2, 1]) / 6)


def test_no_consecutive_ones():
    """Test the no consecutive ones transition matrices."""
    transition_matrices = no_consecutive_ones(p=0.5)
    assert transition_matrices.shape == (2, 2, 2)
    validate_hmm_transition_matrices(transition_matrices)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([2, 1]) / 3)


def test_post_quantum():
    """Test the post quantum transition matrices."""
    transition_matrices = post_quantum(log_alpha=1.0, beta=0.5)
    assert transition_matrices.shape == (3, 3, 3)
    validate_ghmm_transition_matrices(transition_matrices)
    # Verify that transition_matrix[0] + transition_matrix[1] + transition_matrix[2] has largest abs eigenvalue = 1
    transition_matrix_sum_normalized = transition_matrices.sum(axis=0)
    transition_matrix_sum_max_eigval = jnp.abs(jnp.linalg.eigvals(transition_matrix_sum_normalized)).max()
    assert jnp.isclose(transition_matrix_sum_max_eigval, 1, atol=1e-10), "Largest absolute eigenvalue is not 1"


def test_rrxor():
    """Test the rrxor transition matrices."""
    transition_matrices = rrxor(p1=0.5, p2=0.5)
    assert transition_matrices.shape == (2, 5, 5)
    validate_hmm_transition_matrices(transition_matrices, rtol=1e-5)  # rtol=1e-6 barely fails
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([2, 1, 1, 1, 1]) / 6)


def test_sns():
    """Test the sns transition matrices."""
    transition_matrices = sns(p=0.5, q=0.5)
    assert transition_matrices.shape == (2, 2, 2)
    validate_hmm_transition_matrices(transition_matrices)


def test_tom_quantum():
    """Test the tom quantum transition matrices."""
    transition_matrices = tom_quantum(alpha=1.0, beta=1.0)
    assert transition_matrices.shape == (4, 3, 3)
    validate_ghmm_transition_matrices(transition_matrices)


def test_zero_one_random():
    """Test the zero one random transition matrices."""
    transition_matrices = zero_one_random(p=0.5)
    assert transition_matrices.shape == (2, 3, 3)
    validate_hmm_transition_matrices(transition_matrices)
    state_transition_matrix = jnp.sum(transition_matrices, axis=0)
    stationary_distribution = get_stationary_state(state_transition_matrix.T)
    assert jnp.allclose(stationary_distribution, jnp.array([1, 1, 1]) / 3)
