from typing import Any, Protocol

import chex
import optax
from penzai.toolshed.basic_training import LossFunction

from simplexity.predictive_models.predictive_model import PredictiveModel


class Trainer(Protocol):
    """A trainer for a predictive model."""

    @classmethod
    def build(
        cls,
        root_rng: chex.PRNGKey,
        model: PredictiveModel,
        optimizer_def: optax.GradientTransformation,
        loss_fn: LossFunction,
    ) -> "Trainer":
        """Build a trainer."""
        ...

    def step(self, **kwargs) -> Any:
        """Take a step in training."""
        ...
