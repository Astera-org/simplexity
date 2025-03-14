from enum import Enum

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
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


class GHMMProcessType(Enum):
    """The type of generative process to build."""

    DAYS_OF_WEEK = "days_of_week"
    EVEN_ONES = "even_ones"
    FANIZZA = "fanizza"
    MESS3 = "mess3"
    NO_CONSECUTIVE_ONES = "no_consecutive_ones"
    POST_QUANTUM = "post_quantum"
    RRXOR = "rrxor"
    TOM_QUANTUM = "tom_quantum"
    ZERO_ONE_RANDOM = "zero_one_random"


def build_generalized_hidden_markov_model(process_name: str, **kwargs) -> GeneralizedHiddenMarkovModel:
    """Build a generative process from a process type."""
    process_type = GHMMProcessType(process_name)
    if process_type == GHMMProcessType.DAYS_OF_WEEK:
        transition_matrices = days_of_week()
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.EVEN_ONES:
        transition_matrices = even_ones(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.FANIZZA:
        transition_matrices = fanizza(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.MESS3:
        transition_matrices = mess3(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.NO_CONSECUTIVE_ONES:
        transition_matrices = no_consecutive_ones(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.POST_QUANTUM:
        transition_matrices = post_quantum(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.RRXOR:
        transition_matrices = rrxor(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.TOM_QUANTUM:
        transition_matrices = tom_quantum(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    elif process_type == GHMMProcessType.ZERO_ONE_RANDOM:
        transition_matrices = zero_one_random(**kwargs)
        return GeneralizedHiddenMarkovModel(transition_matrices)
    raise KeyError(f"Unknown process type: {process_type}")


class HMMProcessType(Enum):
    """The type of generative process to build."""

    DAYS_OF_WEEK = "days_of_week"
    EVEN_ONES = "even_ones"
    MESS3 = "mess3"
    NO_CONSECUTIVE_ONES = "no_consecutive_ones"
    RRXOR = "rrxor"
    ZERO_ONE_RANDOM = "zero_one_random"


def build_hidden_markov_model(process_name: str, **kwargs) -> HiddenMarkovModel:
    """Build a generative process from a process type."""
    process_type = HMMProcessType(process_name)
    if process_type == HMMProcessType.DAYS_OF_WEEK:
        transition_matrices = days_of_week()
        return HiddenMarkovModel(transition_matrices)
    elif process_type == HMMProcessType.EVEN_ONES:
        transition_matrices = even_ones(**kwargs)
        return HiddenMarkovModel(transition_matrices)
    elif process_type == HMMProcessType.MESS3:
        transition_matrices = mess3(**kwargs)
        return HiddenMarkovModel(transition_matrices)
    elif process_type == HMMProcessType.NO_CONSECUTIVE_ONES:
        transition_matrices = no_consecutive_ones(**kwargs)
        return HiddenMarkovModel(transition_matrices)
    elif process_type == HMMProcessType.RRXOR:
        transition_matrices = rrxor(**kwargs)
        return HiddenMarkovModel(transition_matrices)
    elif process_type == HMMProcessType.ZERO_ONE_RANDOM:
        transition_matrices = zero_one_random(**kwargs)
        return HiddenMarkovModel(transition_matrices)
    raise KeyError(f"Unknown process type: {process_type}")
