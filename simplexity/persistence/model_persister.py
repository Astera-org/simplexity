from typing import Any, Protocol


class ModelPersister(Protocol):
    """Persists a model to a file."""

    def cleanup(self) -> None:
        """Cleans up the persister."""
        ...

    def save_weights(self, model: Any, step: int = 0) -> None:
        """Saves a model."""
        ...

    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Load weights into an existing model instance."""
        ...


def is_model_persister_target(target: str) -> bool:
    """Check if the target is a model persister target."""
    return target.startswith("simplexity.persistence.")
