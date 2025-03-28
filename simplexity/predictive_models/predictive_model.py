from typing import Protocol, runtime_checkable

import jax
from jaxtyping import PyTree


@runtime_checkable
class PredictiveModel(PyTree, Protocol):
    """A predictive model that takes observations and returns a logit distribution over observations."""

    in_size: int
    out_size: int

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predict the next state given the current state."""
        ...
