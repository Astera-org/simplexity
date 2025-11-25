"""Local persister."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LocalPersister(ABC):
    """Persists a model to the local filesystem."""

    directory: Path

    def cleanup(self) -> None:  # noqa: B027
        """Cleans up the persister."""

    @abstractmethod
    def save_weights(self, model: Any, step: int = 0) -> None:
        """Saves a model."""

    @abstractmethod
    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Load weights into an existing model instance."""
