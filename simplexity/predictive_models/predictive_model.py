from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class PredictiveModel(Protocol):
    """A predictive model that takes observations and returns a logit distribution over observations."""

    def __call__(self, x: jax.Array, /) -> jax.Array:
        """Predict the next state given the current state."""
        ...


def is_predictive_model_target(target: str) -> bool:
    """Check if the target is a predictive model target."""
    parts = target.split(".")
    if len(parts) > 2:
        if parts[1] == "nn":  # torch.nn, equinox.nn, penzai.nn
            return True
        if "models" in parts[1]:  # penzai.models, simplexity.predictive_models
            return True
    return parts[0] == "transformer_lens"
