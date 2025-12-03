"""Test the builder module."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import chex
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import (
    add_begin_of_sequence_token,
    build_chain_from_spec,
    build_chain_process,
    build_chain_process_from_spec,
    build_factored_process,
    build_generalized_hidden_markov_model,
    build_hidden_markov_model,
    build_matrices_from_spec,
    build_nonergodic_hidden_markov_model,
    build_nonergodic_initial_state,
    build_nonergodic_transition_matrices,
    build_symmetric_from_spec,
    build_symmetric_process,
    build_symmetric_process_from_spec,
    build_transition_coupled_from_spec,
    build_transition_coupled_process,
    build_transition_coupled_process_from_spec,
    build_transition_matrices,
)
from simplexity.generative_processes.factored_generative_process import FactoredGenerativeProcess
from simplexity.generative_processes.structures import (
    ConditionalTransitions,
    FullyConditional,
    SequentialConditional,
)
from simplexity.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS
from tests.generative_processes.test_transition_matrices import validate_hmm_transition_matrices


def test_build_transition_matrices():
    """Test the build_transition_matrices function."""
    transition_matrices = build_transition_matrices(
        HMM_MATRIX_FUNCTIONS, process_name="coin", process_params={"p": 0.6}
    )
    assert transition_matrices.shape == (2, 1, 1)
    expected = jnp.array([[[0.6]], [[0.4]]])
    chex.assert_trees_all_close(transition_matrices, expected)


def test_add_begin_of_sequence_token():
    """Test the add_begin_of_sequence_token function."""
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
    """Test the build_hidden_markov_model function."""
    hmm = build_hidden_markov_model(process_name="even_ones", process_params={"p": 0.5})
    assert hmm.vocab_size == 2

    with pytest.raises(KeyError):  # noqa: PT011
        build_hidden_markov_model(process_name="fanizza", process_params={"alpha": 2000, "lamb": 0.49})

    with pytest.raises(TypeError):
        build_hidden_markov_model(process_name="even_ones", process_params={"bogus": 0.5})


def test_build_generalized_hidden_markov_model():
    """Test the build_generalized_hidden_markov_model function."""
    ghmm = build_generalized_hidden_markov_model(process_name="even_ones", process_params={"p": 0.5})
    assert ghmm.vocab_size == 2

    ghmm = build_generalized_hidden_markov_model(process_name="fanizza", process_params={"alpha": 2000, "lamb": 0.49})
    assert ghmm.vocab_size == 2

    with pytest.raises(KeyError):  # noqa: PT011
        build_generalized_hidden_markov_model(process_name="dummy")


def test_build_nonergodic_transition_matrices():
    """Test the build_nonergodic_transition_matrices function."""
    coin_1 = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name="coin", process_params={"p": 0.6})
    coin_2 = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name="coin", process_params={"p": 0.3})
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
    """Test the build_nonergodic_initial_state function."""
    state_1 = jnp.array([0.25, 0.40, 0.35])
    state_2 = jnp.array([0.7, 0.3])
    initial_state = build_nonergodic_initial_state([state_1, state_2], jnp.array([0.8, 0.2]))
    assert initial_state.shape == (5,)
    expected = jnp.array([0.20, 0.32, 0.28, 0.14, 0.06])
    chex.assert_trees_all_close(initial_state, expected)


def test_build_nonergodic_hidden_markov_model():
    """Test the build_nonergodic_hidden_markov_model function."""
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["coin", "coin"],
        process_params=[{"p": 0.6}, {"p": 0.3}],
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
    """Test the build_nonergodic_hidden_markov_model function with a nonergodic process."""
    kwargs = {"p": 0.4, "q": 0.25}
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["mr_name", "mr_name"],
        process_params=[kwargs, kwargs],
        process_weights=[0.8, 0.2],
        vocab_maps=[[0, 1, 2, 3], [0, 1, 2, 4]],
        add_bos_token=False,
    )
    assert hmm.vocab_size == 5
    assert hmm.num_states == 8
    assert hmm.transition_matrices.shape == (5, 8, 8)
    validate_hmm_transition_matrices(hmm.transition_matrices, ergodic=False)


def test_build_nonergodic_hidden_markov_model_bos():
    """Test the build_nonergodic_hidden_markov_model function with a BOS token."""
    hmm = build_nonergodic_hidden_markov_model(
        process_names=["coin", "coin"],
        process_params=[{"p": 0.6}, {"p": 0.3}],
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


def _components_spec():
    return [
        {
            "component_type": "hmm",
            "variants": [{"process_name": "coin", "process_params": {"p": 0.6}}],
        },
        {
            "component_type": "hmm",
            "variants": [
                {"process_name": "coin", "process_params": {"p": 0.25}},
                {"process_name": "coin", "process_params": {"p": 0.75}},
            ],
        },
    ]


def _chain_spec():
    spec = _components_spec()
    spec[1] = {**spec[1], "control_map": [0, 1]}
    return spec


def _symmetric_control_maps():
    return [[0, 1], [1, 0]]


def _transition_coupled_inputs():
    return (
        [[0, 1], [1, 0]],
        [0, 1],
        [None, [0, 1]],
    )


def test_build_matrices_from_spec_returns_consistent_arrays():
    """Factored specs should yield aligned parameter shapes."""
    component_types, transition_matrices, normalizing_eigenvectors, initial_states = build_matrices_from_spec(
        _components_spec()
    )
    assert component_types == ["hmm", "hmm"]
    assert transition_matrices[0].shape == (1, 2, 1, 1)
    assert transition_matrices[1].shape[0] == 2
    assert normalizing_eigenvectors[1].shape == (2, transition_matrices[1].shape[2])
    for state in initial_states:
        chex.assert_trees_all_close(jnp.sum(state), jnp.array(1.0))


def test_build_chain_from_spec_returns_control_maps():
    """build_chain_from_spec should return encoded control maps."""
    (
        component_types,
        transition_matrices,
        normalizing_eigenvectors,
        initial_states,
        control_maps,
    ) = build_chain_from_spec(_chain_spec())
    assert component_types == ["hmm", "hmm"]
    assert control_maps[0] is None
    chex.assert_trees_all_close(control_maps[1], jnp.array([0, 1], dtype=jnp.int32))


def test_build_chain_from_spec_missing_control_map_raises():
    """Every non-root node in a chain must provide a control map."""
    with pytest.raises(ValueError):
        build_chain_from_spec(_components_spec())


def test_build_symmetric_from_spec_validates_lengths():
    """Symmetric control maps must cover every combination of other tokens."""
    (
        component_types,
        transition_matrices,
        normalizing_eigenvectors,
        initial_states,
        control_maps,
    ) = build_symmetric_from_spec(_components_spec(), _symmetric_control_maps())
    assert component_types == ["hmm", "hmm"]
    assert len(control_maps) == 2
    chex.assert_trees_all_close(control_maps[0], jnp.array([0, 1], dtype=jnp.int32))
    with pytest.raises(ValueError):
        build_symmetric_from_spec(_components_spec(), [[0], [0, 1]])


def test_build_transition_coupled_from_spec_handles_emission_maps():
    """Transition-coupled specs should surface emission controls when provided."""
    result = build_transition_coupled_from_spec(
        _components_spec(), *_transition_coupled_inputs()
    )
    (
        component_types,
        transition_matrices,
        normalizing_eigenvectors,
        initial_states,
        control_maps_transition,
        emission_variant_indices,
        emission_control_maps,
    ) = result
    assert component_types == ["hmm", "hmm"]
    assert len(control_maps_transition) == 2
    assert emission_variant_indices.shape == (2,)
    assert emission_control_maps is not None
    assert emission_control_maps[0] is None
    chex.assert_trees_all_close(emission_control_maps[1], jnp.array([0, 1], dtype=jnp.int32))


def test_build_chain_process_from_spec_returns_factored_process():
    """High-level chain builder should return a FactoredGenerativeProcess."""
    process = build_chain_process_from_spec(_chain_spec())
    assert isinstance(process, FactoredGenerativeProcess)
    assert isinstance(process.structure, SequentialConditional)
    assert process.vocab_size == 4


def test_build_factored_process_dispatches_topologies():
    """build_factored_process should route to the appropriate topology builder."""
    chain_components = build_chain_from_spec(_chain_spec())
    chain_process = build_factored_process(
        topology_type="chain",
        component_types=chain_components[0],
        transition_matrices=chain_components[1],
        normalizing_eigenvectors=chain_components[2],
        initial_states=chain_components[3],
        control_maps=chain_components[4],
    )
    assert isinstance(chain_process.structure, SequentialConditional)

    symmetric_components = build_symmetric_from_spec(_components_spec(), _symmetric_control_maps())
    symmetric_process = build_factored_process(
        topology_type="symmetric",
        component_types=symmetric_components[0],
        transition_matrices=symmetric_components[1],
        normalizing_eigenvectors=symmetric_components[2],
        initial_states=symmetric_components[3],
        control_maps=symmetric_components[4],
    )
    assert isinstance(symmetric_process.structure, FullyConditional)

    transition_components = build_transition_coupled_from_spec(
        _components_spec(), *_transition_coupled_inputs()
    )
    transition_process = build_factored_process(
        topology_type="transition_coupled",
        component_types=transition_components[0],
        transition_matrices=transition_components[1],
        normalizing_eigenvectors=transition_components[2],
        initial_states=transition_components[3],
        control_maps_transition=transition_components[4],
        emission_variant_indices=transition_components[5],
        emission_control_maps=transition_components[6],
    )
    assert isinstance(transition_process.structure, ConditionalTransitions)
    assert transition_process.vocab_size == 4
    assert transition_process.structure.use_emission_chain is True
