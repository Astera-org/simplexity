from abc import abstractmethod

import equinox as eqx
import jax


class PredictiveModel(eqx.Module):
    """A predictive model that takes observations and returns a logit distribution over observations."""

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """Predict the next state given the current state."""
        ...
