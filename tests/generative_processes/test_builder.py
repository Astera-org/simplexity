import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    add_begin_of_sequence_token,
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
    build_nonergodic_hidden_markov_model,
    build_nonergodic_initial_state,
    build_nonergodic_transition_matrices,
    build_transition_matrices,
)
from simplexity.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS
from tests.generative_processes.test_transition_matrices import validate_hmm_transition_matrices


def test_build_transition_matrices():
    transition_matrices = build_transition_matrices(HMM_MATRIX_FUNCTIONS, "coin", p=0.6)
    assert transition_matrices.shape == (2, 1, 1)
    expected = jnp.array([[[0.6]], [[0.4]]])
    chex.assert_trees_all_close(transition_matrices, expected)


def test_add_begin_of_sequence_token():
    transition_matrix = jnp.array(
        [
            [
                [0.10, 0.20],
                [0.35, 0.25],
            ],
            [
                [0.30, 0.40],
                [0.25, 0.15],
            ],
        ]
    )
    initial_state = jnp.array([0.45, 0.55])
    augmented_matrix = add_begin_of_sequence_token(transition_matrix, initial_state)
    assert augmented_matrix.shape == (3, 3, 3)
    expected = jnp.array(
        [
            [
                [0.10, 0.20, 0.00],
                [0.35, 0.25, 0.00],
                [0.00, 0.00, 0.00],
            ],
            [
                [0.30, 0.40, 0.00],
                [0.25, 0.15, 0.00],
                [0.00, 0.00, 0.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.45, 0.55, 0.00],
            ],
        ]
    )
    chex.assert_trees_all_close(augmented_matrix, expected)


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
        add_bos_token=False,
    )
    assert hmm.vocab_size == 3
    assert hmm.num_states == 2
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
    assert hmm.initial_state.shape == (2,)
    chex.assert_trees_all_close(hmm.initial_state, jnp.array([0.8, 0.2]))


def test_build_nonergodic_hidden_markov_model_with_nonergodic_process():
    kwargs = {"p": 0.4, "q": 0.25}
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["mr_name", "mr_name"],
        process_kwargs=[kwargs, kwargs],
        mixture_weights=jnp.array([0.8, 0.2]),
        vocab_maps=[[0, 1, 2, 3], [0, 1, 2, 4]],
        add_bos_token=False,
    )
    assert hmm.vocab_size == 5
    assert hmm.num_states == 8
    assert hmm.transition_matrices.shape == (5, 8, 8)
    validate_hmm_transition_matrices(hmm.transition_matrices, ergodic=False)


def test_build_nonergodic_hidden_markov_model_bos():
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["coin", "coin"],
        process_kwargs=[{"p": 0.6}, {"p": 0.3}],
        mixture_weights=jnp.array([0.8, 0.2]),
        vocab_maps=[[0, 1], [0, 2]],
        add_bos_token=True,
    )
    assert hmm.vocab_size == 4
    assert hmm.num_states == 3
    expected_transition_matrices = jnp.array(
        [
            [
                [0.6, 0, 0],
                [0, 0.3, 0],
                [0, 0, 0],
            ],
            [
                [0.4, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0.7, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0.8, 0.2, 0],
            ],
        ]
    )
    chex.assert_trees_all_close(hmm.transition_matrices, expected_transition_matrices)
    assert hmm.initial_state.shape == (3,)
    chex.assert_trees_all_close(hmm.initial_state, jnp.array([0, 0, 1.0]))
