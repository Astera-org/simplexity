from abc import abstractmethod

import chex
import equinox as eqx
import jax


class StateSampler(eqx.Module):
    """A sampler for the state of a process."""

    @abstractmethod
    def sample(self, key: chex.PRNGKey) -> jax.Array:
        """Sample a state from the process."""
        ...
