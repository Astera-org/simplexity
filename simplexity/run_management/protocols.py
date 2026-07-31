"""Structural contracts for components that run management instantiates.

A component is identified by what it can do, not by where its code lives. This matters for
generative processes: the reference implementations in
[generators](https://github.com/ealt/generators) are distributed by copying rather than
by import, so a conforming process may be a project-local module that shares no namespace or
base class with simplexity.

The operations mirror the generators specification, which fixes required results rather than
their organization (SPEC.md sections 2 and 6.1). `GenerativeProcess` covers the operations the
runner and the training path exercise; log-space operations are a separate protocol because
only belief-state analysis needs them.
"""

from typing import Any, Protocol, runtime_checkable

import chex
import jax


def _protocol_member_names(protocol: type) -> frozenset[str]:
    """The public member names a protocol declares, including inherited ones.

    Walks the MRO rather than reading the protocol's `__protocol_attrs__`, which is a CPython
    implementation detail that static analysis does not model.
    """
    return frozenset(
        name
        for klass in protocol.__mro__
        if klass is not object and klass.__module__ != "typing"
        for name in vars(klass)
        if not name.startswith("_")
    )


@runtime_checkable
class GenerativeProcess(Protocol):
    """A probabilistic model over observation sequences that run management can drive.

    Implementations may be simplexity classes, subclasses of
    `simplexity.generative_processes.generative_process.GenerativeProcess`, or project-local
    modules vendored from generators and adapted. Only the members below are required.
    """

    @property
    def vocab_size(self) -> int:
        """The number of observations that can be emitted by the generative process."""
        ...

    @property
    def initial_state(self) -> Any:
        """The initial state of the generative process."""
        ...

    def emit_observation(self, state: Any, key: chex.PRNGKey) -> chex.Array:
        """Emit an observation based on the state of the generative process."""
        ...

    def transition_states(self, state: Any, obs: chex.Array) -> Any:
        """Evolve the state of the generative process based on the observation."""
        ...

    def observation_probability_distribution(self, state: Any) -> jax.Array:
        """Compute the distribution over observations that can be emitted from a state."""
        ...

    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        ...

    def generate(
        self, state: Any, key: chex.PRNGKey, sequence_len: int, return_all_states: bool
    ) -> tuple[Any, chex.Array]:
        """Generate a batch of observation sequences, optionally returning every belief state."""
        ...


@runtime_checkable
class LogSpaceGenerativeProcess(GenerativeProcess, Protocol):
    """A generative process that also exposes log-space operations.

    Required only by belief-state analysis, such as mixed-state presentation, where working in
    log space keeps long-sequence probabilities numerically stable. The training and generation
    paths never call these.
    """

    def log_observation_probability_distribution(self, log_belief_state: Any) -> jax.Array:
        """Compute the log distribution over observations that can be emitted from a state."""
        ...

    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        ...


GENERATIVE_PROCESS_MEMBERS = _protocol_member_names(GenerativeProcess)
LOG_SPACE_GENERATIVE_PROCESS_MEMBERS = _protocol_member_names(LogSpaceGenerativeProcess)


def missing_generative_process_members(obj: object) -> list[str]:
    """List the `GenerativeProcess` members that an object does not provide.

    `isinstance` against a runtime-checkable protocol reports only whether an object conforms.
    Reporting *which* members are absent turns a rejected config into an actionable error.

    Args:
        obj: The candidate generative process.

    Returns:
        The names of missing members, sorted.
    """
    return sorted(name for name in GENERATIVE_PROCESS_MEMBERS if not hasattr(obj, name))
