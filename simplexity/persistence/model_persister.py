from abc import abstractmethod

import equinox as eqx

from simplexity.predictive_models.predictive_model import PredictiveModel


class ModelPersister:
    """Persists a model to a file."""

    @abstractmethod
    def save_weights(self, model: PredictiveModel, name: str) -> None:
        """Saves a model."""
        ...

    @abstractmethod
    def load_weights(self, model: PredictiveModel, name: str) -> PredictiveModel:
        """Load weights into an existing model instance."""
        ...
