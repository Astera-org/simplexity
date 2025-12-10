"""Local model persister protocol."""

from typing import Any, Protocol


class LocalModelPersister(Protocol):
    """Persists a model to a local directory."""

    directory: Any
    """Return the directory where the model is persisted."""

    def cleanup(self) -> None:
        """Cleans up the persister."""
        ...

    def save_weights(self, model: Any, step: int = 0) -> None:
        """Saves a model."""
        ...

    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Load weights into an existing model instance."""
        ...
