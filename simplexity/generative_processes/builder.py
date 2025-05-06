import inspect
from collections.abc import Callable

import jax

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import GHMM_MATRIX_FUNCTIONS, HMM_MATRIX_FUNCTIONS


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


def build_hidden_markov_model(process_name: str, initial_state: jax.Array | None = None, **kwargs) -> HiddenMarkovModel:
    """Build a hidden Markov model."""
    transition_matrices = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, **kwargs)
    return HiddenMarkovModel(transition_matrices, initial_state)


def build_generalized_hidden_markov_model(process_name: str, **kwargs) -> GeneralizedHiddenMarkovModel:
    """Build a generalized hidden Markov model."""
    transition_matrices = build_transition_matrices(GHMM_MATRIX_FUNCTIONS, process_name, **kwargs)
    return GeneralizedHiddenMarkovModel(transition_matrices)
