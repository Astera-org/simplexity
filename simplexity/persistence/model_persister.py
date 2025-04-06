from abc import abstractmethod
from typing import Self

import equinox as eqx

from simplexity.predictive_models.predictive_model import PredictiveModel


class ModelPersister(eqx.Module):
    """Persists a model to a file."""

    def __enter__(self, *args, **kwargs) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Cleans up the persister."""
        ...

    @abstractmethod
    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Saves a model."""
        ...

    @abstractmethod
    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Load weights into an existing model instance."""
        ...
