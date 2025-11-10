from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LocalPersister(ABC):
    """Persists a model to the local filesystem."""

    directory: Path

    def cleanup(self) -> None:
        """Cleans up the persister."""
        ...

    @abstractmethod
    def save_weights(self, model: Any, step: int = 0) -> None:
        """Saves a model."""
        ...

    @abstractmethod
    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Load weights into an existing model instance."""
        ...
