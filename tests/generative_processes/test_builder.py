import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
    build_nonergodic_hidden_markov_model,
    build_nonergodic_initial_state,
    build_nonergodic_transition_matrices,
    build_transition_matrices,
)
from simplexity.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS


def test_build_transition_matrices():
    transition_matrices = build_transition_matrices(HMM_MATRIX_FUNCTIONS, "coin", p=0.6)
    assert transition_matrices.shape == (2, 1, 1)
    expected = jnp.array([[[0.6]], [[0.4]]])
    chex.assert_trees_all_close(transition_matrices, expected)


def test_build_hidden_markov_model():
    hmm = build_hidden_markov_model("even_ones", p=0.5)
    assert hmm.vocab_size == 2

    with pytest.raises(KeyError):  # noqa: PT011
        build_hidden_markov_model("fanizza", alpha=2000, lamb=0.49)

    with pytest.raises(TypeError):
        build_hidden_markov_model("even_ones", bogus=0.5)


def test_build_generalized_hidden_markov_model():
    ghmm = build_generalized_hidden_markov_model("even_ones", p=0.5)
    assert ghmm.vocab_size == 2

    ghmm = build_generalized_hidden_markov_model("fanizza", alpha=2000, lamb=0.49)
    assert ghmm.vocab_size == 2

    with pytest.raises(KeyError):  # noqa: PT011
        build_generalized_hidden_markov_model("dummy")


def test_build_nonergodic_transition_matrices():
    coin_1 = build_transition_matrices(HMM_MATRIX_FUNCTIONS, "coin", p=0.6)
    coin_2 = build_transition_matrices(HMM_MATRIX_FUNCTIONS, "coin", p=0.3)
    transition_matrices = build_nonergodic_transition_matrices([coin_1, coin_2], [[0, 1], [0, 2]])
    assert transition_matrices.shape == (3, 2, 2)
    expected = jnp.array(
        [
            [
                [0.6, 0],
                [0, 0.3],
            ],
            [
                [0.4, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0.7],
            ],
        ]
    )
    chex.assert_trees_all_close(transition_matrices, expected)


def test_build_nonergodic_initial_state():
    state_1 = jnp.array([0.25, 0.40, 0.35])
    state_2 = jnp.array([0.7, 0.3])
    initial_state = build_nonergodic_initial_state([state_1, state_2], jnp.array([0.8, 0.2]))
    assert initial_state.shape == (5,)
    expected = jnp.array([0.20, 0.32, 0.28, 0.14, 0.06])
    chex.assert_trees_all_close(initial_state, expected)


def test_build_nonergodic_hidden_markov_model():
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["coin", "coin"],
        process_kwargs=[{"p": 0.6}, {"p": 0.3}],
        mixture_weights=jnp.array([0.8, 0.2]),
        vocab_maps=[[0, 1], [0, 2]],
    )
    assert hmm.vocab_size == 3
    assert hmm.num_states == 2
    assert hmm.initial_state.shape == (2,)
    expected_transition_matrices = jnp.array(
        [
            [
                [0.6, 0],
                [0, 0.3],
            ],
            [
                [0.4, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0.7],
            ],
        ]
    )
    chex.assert_trees_all_close(hmm.transition_matrices, expected_transition_matrices)
    expected_initial_state = jnp.array([0.8, 0.2])
    chex.assert_trees_all_close(hmm.initial_state, expected_initial_state)
