import inspect

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import (
    ALL_GHMMS,
    ALL_HMMS,
    GHMMProcessType,
    HMMProcessType,
)


def build_hidden_markov_model(process_name: str, **kwargs) -> HiddenMarkovModel:
    """Build an HMM from a process type."""
    if process_name not in HMMProcessType:
        raise KeyError(
            f'Unknown process type: "{process_name}".  '
            f"Available HMM processes are: {', '.join(p.value for p in HMMProcessType)}"
        )
    func = ALL_HMMS.get(HMMProcessType(process_name))
    assert func is not None, f"HMM not defined for {process_name}"
    sig = inspect.signature(func)
    try:
        sig.bind_partial(**kwargs)
        return HiddenMarkovModel(func(**kwargs))
    except TypeError as e:
        params = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(f"Invalid arguments for {process_name}: {e}.  Signature is: {params}") from e


def build_generalized_hidden_markov_model(process_name: str, **kwargs) -> GeneralizedHiddenMarkovModel:
    """Build a generalized HMM from a process type."""
    if process_name not in GHMMProcessType:
        raise KeyError(
            f'Unknown process type: "{process_name}".  '
            f"Available GHMM processes are: {', '.join(p.value for p in GHMMProcessType)}"
        )
    func = ALL_GHMMS.get(GHMMProcessType(process_name))
    assert func is not None, f"GHMM not defined for {process_name}"
    sig = inspect.signature(func)
    try:
        sig.bind_partial(**kwargs)
        return GeneralizedHiddenMarkovModel(func(**kwargs))
    except TypeError as e:
        params = ", ".join(f"{k}: {v.annotation}" for k, v in sig.parameters.items())
        raise TypeError(f"Invalid arguments for {process_name}: {e}.  Signature is {params}") from e
