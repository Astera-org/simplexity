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

    # --- Checkpoint discovery helpers ---
    @abstractmethod
    def list_checkpoints(self) -> list[int]:
        """List all available checkpoint steps in this persister."""
        ...

    @abstractmethod
    def latest_checkpoint(self) -> int | None:
        """Return the latest checkpoint step, or None if none exist."""
        ...

    @abstractmethod
    def checkpoint_exists(self, step: int) -> bool:
        """Return True if a checkpoint exists for the given step."""
        ...

    def uri_for_step(self, step: int) -> str:
        """Return a URI for the checkpoint at the given step.

        Subclasses should override this to provide a stable artifact URI
        (e.g., file://, s3://). Default implementation returns an empty string.
        """
        return ""
