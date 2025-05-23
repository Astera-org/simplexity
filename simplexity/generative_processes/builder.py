import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import (
    GHMM_MATRIX_FUNCTIONS,
    HMM_MATRIX_FUNCTIONS,
    stationary_state,
)


def build_transition_matrices(matrix_functions: dict[str, Callable], process_name: str, **kwargs) -> jax.Array:
    """Build transition matrices for a generative process."""
    if process_name not in matrix_functions:
        raise KeyError(
            f'Unknown process type: "{process_name}".  '
            f"Available HMM processes are: {', '.join(matrix_functions.keys())}"
        )
    matrix_function = matrix_functions[process_name]
    sig = inspect.signature(matrix_function)
    try:
        sig.bind_partial(**kwargs)
        transition_matrices = matrix_function(**kwargs)
    except TypeError as e:
        params = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(f"Invalid arguments for {process_name}: {e}.  Signature is: {params}") from e
    return transition_matrices


def add_begin_of_sequence_token(transition_matrix: jax.Array, initial_state: jax.Array | None = None) -> jax.Array:
    """Augments transition matrices with a BOS token."""
    vocab_size, num_states, _ = transition_matrix.shape
    augmented_matrix = jnp.zeros((vocab_size + 1, num_states + 1, num_states + 1), dtype=transition_matrix.dtype)
    augmented_matrix = augmented_matrix.at[:vocab_size, :num_states, :num_states].set(transition_matrix)
    if initial_state is None:
        initial_state = stationary_state(transition_matrix.sum(axis=0).T)
    return augmented_matrix.at[vocab_size, num_states, :num_states].set(initial_state)


def build_hidden_markov_model(process_name: str, initial_state: jax.Array | None = None, **kwargs) -> HiddenMarkovModel:
    """Build a hidden Markov model."""
    transition_matrices = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, **kwargs)
    return HiddenMarkovModel(transition_matrices, initial_state)


def build_generalized_hidden_markov_model(process_name: str, **kwargs) -> GeneralizedHiddenMarkovModel:
    """Build a generalized hidden Markov model."""
    transition_matrices = build_transition_matrices(GHMM_MATRIX_FUNCTIONS, process_name, **kwargs)
    return GeneralizedHiddenMarkovModel(transition_matrices)


def build_nonergodic_transition_matrices(
    component_transition_matrices: Sequence[jax.Array], vocab_maps: Sequence[Sequence[int]] | None = None
) -> jax.Array:
    """Build composite transition matrices of a nonergodic process from component transition matrices."""
    if vocab_maps is None:
        vocab_maps = [list(range(matrix.shape[0])) for matrix in component_transition_matrices]
    vocab_size = max(max(vocab_map) for vocab_map in vocab_maps) + 1
    total_states = sum(matrix.shape[1] for matrix in component_transition_matrices)
    composite_transition_matrix = jnp.zeros((vocab_size, total_states, total_states))
    state_offset = 0
    for matrix, vocab_map in zip(component_transition_matrices, vocab_maps, strict=True):
        for component_vocab_idx, composite_vocab_idx in enumerate(vocab_map):
            composite_transition_matrix = composite_transition_matrix.at[
                composite_vocab_idx,
                state_offset : state_offset + matrix.shape[1],
                state_offset : state_offset + matrix.shape[1],
            ].set(matrix[component_vocab_idx])
        state_offset += matrix.shape[1]
    return composite_transition_matrix


def build_nonergodic_initial_state(
    component_initial_states: Sequence[jax.Array], process_weights: jax.Array
) -> jax.Array:
    """Build initial state for a nonergodic process from component initial states."""
    assert process_weights.shape == (len(component_initial_states),)
    assert jnp.all(process_weights >= 0)
    process_probabilities = process_weights / process_weights.sum()
    return jnp.concatenate(
        [p * state for p, state in zip(process_probabilities, component_initial_states, strict=True)], axis=0
    )


def build_nonergodic_hidden_markov_model(
    process_names: list[str],
    process_kwargs: Sequence[Mapping[str, Any]],
    process_weights: jax.Array,
    vocab_maps: Sequence[Sequence[int]] | None = None,
    add_bos_token: bool = False,
) -> HiddenMarkovModel:
    """Build a hidden Markov model from a list of process names and their corresponding keyword arguments."""
    component_transition_matrices = [
        build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, **kwargs)
        for process_name, kwargs in zip(process_names, process_kwargs, strict=True)
    ]
    composite_transition_matrix = build_nonergodic_transition_matrices(component_transition_matrices, vocab_maps)
    component_initial_states = [
        stationary_state(transition_matrix.sum(axis=0).T) for transition_matrix in component_transition_matrices
    ]
    initial_state = build_nonergodic_initial_state(component_initial_states, process_weights)
    if add_bos_token:
        composite_transition_matrix = add_begin_of_sequence_token(composite_transition_matrix, initial_state)
        num_states = composite_transition_matrix.shape[1]
        initial_state = jnp.zeros((num_states,), dtype=composite_transition_matrix.dtype)
        initial_state = initial_state.at[num_states - 1].set(1)
    return HiddenMarkovModel(composite_transition_matrix, initial_state)
