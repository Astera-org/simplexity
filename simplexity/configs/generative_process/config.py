from dataclasses import dataclass
from typing import Literal

ProcessName = Literal[
    "days_of_week",
    "even_ones",
    "fanizza",
    "mess3",
    "no_consecutive_ones",
    "post_quantum",
    "rrxor",
    "tom_quantum",
    "zero_one_random",
]

ProcessBuilder = Literal[
    "simplexity.generative_processes.builder.build_generalized_hidden_markov_model",
    "simplexity.generative_processes.builder.build_hidden_markov_model",
]
ProcessType = ProcessName


@dataclass
class ProcessInstanceConfig:
    """Configuration for the generative process."""

    _target_: ProcessBuilder
    process_name: ProcessType


@dataclass
class DaysOfWeekConfig(ProcessInstanceConfig):
    """Configuration for DaysOfWeek model."""

    # _target_: build_hidden_markov_model
    # process_name: "days_of_week"


@dataclass
class EvenOnesConfig(ProcessInstanceConfig):
    """Configuration for EvenOnes model."""

    # _target_: build_hidden_markov_model
    # process_name: "even_ones"
    p: float


@dataclass
class FanizzaConfig(ProcessInstanceConfig):
    """Configuration for Fanizza model."""

    # _target_: build_generalized_hidden_markov_model
    # process_name: "fanizza"
    alpha: float
    lamb: float


@dataclass
class Mess3Config(ProcessInstanceConfig):
    """Configuration for Mess3 model."""

    # _target_: build_hidden_markov_model
    # process_name: "mess3"
    x: float
    a: float


@dataclass
class NoConsecutiveOnesConfig(ProcessInstanceConfig):
    """Configuration for NoConsecutiveOnes model."""

    # _target_: build_hidden_markov_model
    # process_name: "no_consecutive_ones"
    p: float


@dataclass
class PostQuantumConfig(ProcessInstanceConfig):
    """Configuration for PostQuantum model."""

    # _target_: build_generalized_hidden_markov_model
    # process_name: "post_quantum"
    log_alpha: float
    beta: float


@dataclass
class RRXorConfig(ProcessInstanceConfig):
    """Configuration for RRXor model."""

    # _target_: build_hidden_markov_model
    # process_name: "rrxor"
    pR1: float
    pR2: float


@dataclass
class TomQuantumConfig(ProcessInstanceConfig):
    """Configuration for TomQuantum model."""

    # _target_: build_generalized_hidden_markov_model
    # process_name: "tom_quantum"
    alpha: float
    beta: float


@dataclass
class ZeroOneRandomConfig(ProcessInstanceConfig):
    """Configuration for ZeroOneRandom model."""

    # _target_: build_hidden_markov_model
    # process_name: "zero_one_random"
    p: float


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: ProcessName
    vocab_size: int
    instance: ProcessInstanceConfig
