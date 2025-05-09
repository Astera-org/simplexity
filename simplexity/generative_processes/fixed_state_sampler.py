import chex
import jax

from simplexity.generative_processes.state_sampler import StateSampler


class FixedStateSampler(StateSampler):
    """A sampler for a fixed state."""

    states: jax.Array

    def __init__(self, state: jax.Array):
        self.state = state

    def sample(self, key: chex.PRNGKey) -> jax.Array:
        """Return the fixed state."""
        return self.state
