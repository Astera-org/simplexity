"""Tests for factoring utilities."""


import chex
import jax.numpy as jnp
import pytest

from simplexity.utils.factoring_utils import (
    TokenEncoder,
    compute_obs_dist_for_variant,
    transition_with_obs,
)


def test_compute_obs_dist_for_variant_ghmm_missing_eigenvector():
    """GHMM without normalizing eigenvector should raise ValueError."""
    state = jnp.array([0.5, 0.5])
    transition_matrix = jnp.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="GHMM requires normalizing_eigenvector"):
        compute_obs_dist_for_variant("ghmm", state, transition_matrix, normalizing_eigenvector=None)

def test_transition_with_obs_ghmm_missing_eigenvector():
    """GHMM transition without normalizing eigenvector should raise ValueError."""
    state = jnp.array([0.5, 0.5])
    transition_matrix = jnp.eye(2)[None, :, :]  # Shape: [V=1, S=2, S=2]
    obs = jnp.array(0)

    with pytest.raises(ValueError, match="GHMM requires normalizing_eigenvector"):
        transition_with_obs("ghmm", state, transition_matrix, obs, normalizing_eigenvector=None)

def test_token_encoder_extract_factors_vectorized():
    """TokenEncoder should handle batch decoding."""
    vocab_sizes = jnp.array([2, 3, 4])
    encoder = TokenEncoder(vocab_sizes)

    # Encode multiple tokens
    tokens = jnp.array([0, 5, 10, 23])  # Multiple composite tokens
    factors = encoder.extract_factors_vectorized(tokens)

    assert factors.shape == (4, 3)  # (batch, num_factors)
    # Verify each decoding is correct
    for i, token in enumerate(tokens):
        expected_tuple = encoder.token_to_tuple(token)
        expected_array = jnp.array([t.item() for t in expected_tuple])
        chex.assert_trees_all_close(factors[i], expected_array)


def test_compute_obs_dist_for_variant_hmm():
    """HMM observation distribution should work without normalizing eigenvector."""
    state = jnp.array([0.6, 0.4])
    # Transition matrix: [V=2, S=2, S=2]
    transition_matrix = jnp.array(
        [
            [[0.8, 0.2], [0.3, 0.7]],  # For obs=0
            [[0.1, 0.9], [0.4, 0.6]],  # For obs=1
        ]
    )

    dist = compute_obs_dist_for_variant("hmm", state, transition_matrix, normalizing_eigenvector=None)

    assert dist.shape == (2,)  # V=2
    # P(obs=0) = state @ transition_matrix[0] @ 1 = [0.6, 0.4] @ [[0.8, 0.2], [0.3, 0.7]] @ [1, 1]
    expected_0 = jnp.sum(state @ transition_matrix[0])
    expected_1 = jnp.sum(state @ transition_matrix[1])
    chex.assert_trees_all_close(dist, jnp.array([expected_0, expected_1]))


def test_transition_with_obs_hmm():
    """HMM transition should work without normalizing eigenvector."""
    state = jnp.array([0.6, 0.4])
    transition_matrix = jnp.array(
        [
            [[0.8, 0.2], [0.3, 0.7]],  # For obs=0
            [[0.1, 0.9], [0.4, 0.6]],  # For obs=1
        ]
    )
    obs = jnp.array(0)

    new_state = transition_with_obs("hmm", state, transition_matrix, obs, normalizing_eigenvector=None)

    assert new_state.shape == (2,)
    # New state should be state @ transition_matrix[0] normalized
    unnormalized = state @ transition_matrix[0]
    expected = unnormalized / jnp.sum(unnormalized)
    chex.assert_trees_all_close(new_state, expected)


def test_token_encoder_tuple_to_token():
    """TokenEncoder should encode tuples to composite tokens."""
    vocab_sizes = jnp.array([2, 3, 4])
    encoder = TokenEncoder(vocab_sizes)

    # Test a few tuples
    tuple0 = (jnp.array(0), jnp.array(0), jnp.array(0))
    tuple1 = (jnp.array(1), jnp.array(2), jnp.array(3))

    token0 = encoder.tuple_to_token(tuple0)
    token1 = encoder.tuple_to_token(tuple1)

    assert token0 == 0  # (0, 0, 0) -> 0
    # (1, 2, 3) -> 1 + 2*2 + 3*2*3 = 1 + 4 + 18 = 23
    assert token1 == 23

    # Verify roundtrip
    decoded0 = encoder.token_to_tuple(token0)
    decoded1 = encoder.token_to_tuple(token1)

    chex.assert_trees_all_close(decoded0, tuple0)
    chex.assert_trees_all_close(decoded1, tuple1)
