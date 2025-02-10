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

    @eqx.filter_vmap(in_axes=(None, 0, 0, None))
    def generate(self, state: State, key: chex.PRNGKey, sequence_len: int) -> tuple[State, chex.Array]:
        """Generate a batch of sequences of observations from the generative process.

        Returns:
            A tuple of (final_states, observations) where:
            - observations has shape (batch_size, sequence_len, observation_dim)
            - final_states has shape (batch_size,) + state_shape
        """
        keys = jax.random.split(key, sequence_len)
        
        def scan_fn(carry, key):
            return self.transition(carry, key)
        
        return jax.lax.scan(scan_fn, state, keys)

    @abstractmethod
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        ...

    @abstractmethod
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        ...
