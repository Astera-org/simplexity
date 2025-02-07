from abc import abstractmethod
from typing import Generic, TypeVar

import chex
import equinox as eqx
import jax

State = TypeVar("State")


class GenerativeProcess(eqx.Module, Generic[State]):
    """A generative process is a probabilistic model that can be used to generate data."""

    @abstractmethod
    def transition(self, state: State, key: chex.PRNGKey) -> tuple[State, chex.Array]:
        """Perform a state transition of the generative process and emit an observation."""
        ...

    @abstractmethod
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        ...

    @abstractmethod
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        ...
