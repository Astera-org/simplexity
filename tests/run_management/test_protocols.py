"""Tests for the structural component contracts."""

import jax.numpy as jnp
import pytest

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.generative_process import GenerativeProcess as GenerativeProcessBaseClass
from simplexity.run_management.protocols import (
    GenerativeProcess,
    LogSpaceGenerativeProcess,
    missing_generative_process_members,
)

CORE_MEMBERS = {
    "emit_observation",
    "generate",
    "initial_state",
    "observation_probability_distribution",
    "probability",
    "transition_states",
    "vocab_size",
}
LOG_SPACE_MEMBERS = {"log_observation_probability_distribution", "log_probability"}


class MinimalProcess:
    """The smallest object satisfying the core protocol."""

    vocab_size = 2
    initial_state = None

    def emit_observation(self, state, key):
        """Emit an observation."""

    def transition_states(self, state, obs):
        """Transition states."""

    def observation_probability_distribution(self, state):
        """Observation distribution."""

    def probability(self, observations):
        """Sequence probability."""

    def generate(self, state, key, sequence_len, return_all_states):
        """Generate sequences."""


class LogSpaceProcess(MinimalProcess):
    """A core process that also provides log-space operations."""

    def log_observation_probability_distribution(self, log_belief_state):
        """Log observation distribution."""

    def log_probability(self, observations):
        """Log sequence probability."""


def test_core_protocol_members_are_pinned() -> None:
    """Guard the contract against accidental widening or narrowing.

    Asserted through the public helper rather than the protocol's internals, so the pin does not
    depend on a CPython implementation detail.
    """
    assert set(missing_generative_process_members(object())) == CORE_MEMBERS


def test_log_space_operations_are_not_in_the_core_contract() -> None:
    """Log-space operations stay optional because only belief-state analysis needs them."""
    assert not LOG_SPACE_MEMBERS & CORE_MEMBERS
    assert isinstance(MinimalProcess(), GenerativeProcess)


@pytest.mark.parametrize("member", sorted(LOG_SPACE_MEMBERS))
def test_omitting_any_log_space_member_breaks_the_extension_only(member: str) -> None:
    """Each log-space member is required by the extension and irrelevant to the core."""
    attrs = {name: getattr(LogSpaceProcess, name) for name in CORE_MEMBERS | LOG_SPACE_MEMBERS if name != member}
    partial_process = type("PartialLogSpaceProcess", (), attrs)()
    assert isinstance(partial_process, GenerativeProcess)
    assert not isinstance(partial_process, LogSpaceGenerativeProcess)


def test_log_space_protocol_requires_the_core_members_too() -> None:
    """The extension is the core contract plus log-space, not log-space alone."""
    attrs = {name: getattr(LogSpaceProcess, name) for name in LOG_SPACE_MEMBERS}
    log_space_only = type("LogSpaceOnly", (), attrs)()
    assert not isinstance(log_space_only, LogSpaceGenerativeProcess)


def test_minimal_object_satisfies_the_core_protocol() -> None:
    """Conformance requires the operations, not a base class."""
    assert isinstance(MinimalProcess(), GenerativeProcess)
    assert not isinstance(MinimalProcess(), GenerativeProcessBaseClass)


def test_core_process_does_not_satisfy_the_log_space_protocol() -> None:
    assert not isinstance(MinimalProcess(), LogSpaceGenerativeProcess)


def test_log_space_process_satisfies_both_protocols() -> None:
    process = LogSpaceProcess()
    assert isinstance(process, GenerativeProcess)
    assert isinstance(process, LogSpaceGenerativeProcess)


def test_simplexity_processes_satisfy_both_protocols() -> None:
    """Relaxing the contract must not exclude the processes implemented in simplexity."""
    process = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
    assert isinstance(process, GenerativeProcess)
    assert isinstance(process, LogSpaceGenerativeProcess)


@pytest.mark.parametrize("member", sorted(CORE_MEMBERS))
def test_omitting_any_core_member_breaks_conformance(member: str) -> None:
    """Every member of the core contract is load-bearing, and the absent one is named."""
    attrs = {name: getattr(MinimalProcess, name) for name in CORE_MEMBERS if name != member}
    partial_process = type("PartialProcess", (), attrs)()
    assert not isinstance(partial_process, GenerativeProcess)
    assert missing_generative_process_members(partial_process) == [member]


def test_missing_members_are_empty_for_a_conforming_process() -> None:
    assert missing_generative_process_members(MinimalProcess()) == []


def test_missing_members_are_reported_sorted() -> None:
    """Reporting which members are absent is what makes a rejected config actionable."""
    assert missing_generative_process_members(object()) == sorted(CORE_MEMBERS)


def test_missing_members_for_a_partially_conforming_object() -> None:
    class HalfProcess:
        vocab_size = 2
        initial_state = jnp.zeros(2)

    assert missing_generative_process_members(HalfProcess()) == sorted(CORE_MEMBERS - {"vocab_size", "initial_state"})
