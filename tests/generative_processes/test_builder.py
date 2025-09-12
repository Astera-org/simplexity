import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    add_begin_of_sequence_token,
    build_factored_generator,
    build_factored_hmm_generator,
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
        process_weights=[0.8, 0.2],
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
        process_weights=[0.8, 0.2],
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
        process_weights=[0.8, 0.2],
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


def test_build_factored_generator():
    """Test building factored generator with mixed component types."""
    # Test 3-component factored generator: mess3(3) + tom_quantum(4) + coin(2) = 24 vocab
    factored_gen = build_factored_generator(
        [
            {"process_name": "mess3", "x": 0.5, "a": 0.8},
            {"process_name": "tom_quantum", "alpha": 0.3, "beta": 0.7},
            {"process_name": "zero_one_random", "p": 0.9},
        ],
        component_types=["hmm", "ghmm", "hmm"],
    )

    assert factored_gen.vocab_size == 24  # 3 * 4 * 2
    assert len(factored_gen.components) == 3

    # Test initial state is tuple of component initial states
    initial_state = factored_gen.initial_state
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 3


def test_build_factored_generator_default_component_types():
    """Test factored generator defaults to GHMM components when component_types=None."""
    factored_gen = build_factored_generator(
        [{"process_name": "zero_one_random", "p": 0.7}, {"process_name": "mess3", "x": 0.4, "a": 0.6}]
    )  # component_types=None should default to ["ghmm", "ghmm"]

    assert factored_gen.vocab_size == 6  # 2 * 3
    assert len(factored_gen.components) == 2


def test_build_factored_generator_errors():
    """Test error handling in build_factored_generator."""
    # Test mismatched component_specs and component_types lengths
    with pytest.raises(ValueError, match="component_specs and component_types must have the same length"):
        build_factored_generator(
            [{"process_name": "zero_one_random", "p": 0.5}], component_types=["hmm", "ghmm"]
        )  # 1 spec, 2 types

    # Test invalid component type
    with pytest.raises(ValueError, match="Unknown component type"):
        build_factored_generator([{"process_name": "zero_one_random", "p": 0.5}], component_types=["invalid_type"])

    # Test invalid process name (should propagate from underlying builders)
    with pytest.raises(KeyError):
        build_factored_generator([{"process_name": "nonexistent_process"}])


def test_build_factored_hmm_generator():
    """Test building factored generator with all HMM components."""
    # Test 3-component HMM factored generator: coin(2) + mess3(3) + coin(2) = 12 vocab
    factored_gen = build_factored_hmm_generator(
        [
            {"process_name": "zero_one_random", "p": 0.6},
            {"process_name": "mess3", "x": 0.3, "a": 0.7},
            {"process_name": "zero_one_random", "p": 0.4},
        ]
    )

    assert factored_gen.vocab_size == 12  # 2 * 3 * 2
    assert len(factored_gen.components) == 3

    # Verify all components are HMMs (they should have transition_matrices attribute)
    for component in factored_gen.components:
        assert hasattr(component, "transition_matrices")
        assert hasattr(component, "initial_state")


def test_build_factored_hmm_generator_with_various_processes():
    """Test factored HMM generator with different process types."""
    factored_gen = build_factored_hmm_generator(
        [
            {"process_name": "even_ones", "p": 0.3},  # vocab=2
            {"process_name": "no_consecutive_ones", "p": 0.7},  # vocab=2
        ]
    )

    assert factored_gen.vocab_size == 4  # 2 * 2
    assert len(factored_gen.components) == 2


def test_factored_generator_hydra_compatibility():
    """Test that factored generators work with Hydra-style parameters."""
    # Test that process_name and extra kwargs are ignored (Hydra compatibility)
    factored_gen = build_factored_hmm_generator(
        [{"process_name": "zero_one_random", "p": 0.5}],
        process_name="ignored_name",  # Should be ignored
        extra_param="ignored_value",  # Should be ignored
    )

    assert factored_gen.vocab_size == 2
    assert len(factored_gen.components) == 1

    # Test with build_factored_generator too
    factored_gen2 = build_factored_generator(
        [{"process_name": "mess3", "x": 0.5, "a": 0.8}],
        component_types=["hmm"],
        _process_name="ignored",  # Underscore prefix convention
        extra_kwarg="also_ignored",
    )

    assert factored_gen2.vocab_size == 3
    assert len(factored_gen2.components) == 1


def test_factored_generator_vocab_size_calculations():
    """Test vocab size calculations for various component combinations."""
    test_cases = [
        # (component_specs, component_types, expected_vocab_size)
        ([{"process_name": "zero_one_random", "p": 0.5}], ["hmm"], 2),  # Single component
        (
            [{"process_name": "zero_one_random", "p": 0.5}, {"process_name": "zero_one_random", "p": 0.7}],
            ["hmm", "hmm"],
            4,
        ),  # 2*2
        (
            [{"process_name": "mess3", "x": 0.5, "a": 0.8}, {"process_name": "zero_one_random", "p": 0.5}],
            ["hmm", "hmm"],
            6,
        ),  # 3*2
        ([{"process_name": "tom_quantum", "alpha": 0.3, "beta": 0.7}], ["ghmm"], 4),  # Single GHMM
        (
            [{"process_name": "mess3", "x": 0.5, "a": 0.8}, {"process_name": "tom_quantum", "alpha": 0.3, "beta": 0.7}],
            ["hmm", "ghmm"],
            12,
        ),  # 3*4
    ]

    for component_specs, component_types, expected_vocab_size in test_cases:
        factored_gen = build_factored_generator(component_specs, component_types)
        assert factored_gen.vocab_size == expected_vocab_size, (
            f"Expected vocab_size {expected_vocab_size}, got {factored_gen.vocab_size} for specs {component_specs}"
        )
