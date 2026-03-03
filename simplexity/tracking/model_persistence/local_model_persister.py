"""Local model persister protocol."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LocalModelPersister(ABC):
    """Abstract base class for local model persisters."""

    directory: Path
    """Return the directory where the model is persisted."""

    def cleanup(self) -> None:  # noqa: B027
        """Cleans up the persister."""

    @abstractmethod
    def save_weights(self, model: Any, step: int = 0) -> None:
        """Saves a model."""

    @abstractmethod
    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Load weights into an existing model instance."""
