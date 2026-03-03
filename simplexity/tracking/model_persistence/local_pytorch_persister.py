"""Local PyTorch persister."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import torch

from simplexity.tracking.model_persistence.local_model_persister import (
    LocalModelPersister,
)


class LocalPytorchPersister(LocalModelPersister):
    """Persists a PyTorch model to the local filesystem."""

    filename: str = "model.pt"

    def __init__(self, directory: str | Path, filename: str = "model.pt"):
        self.directory = Path(directory)
        self.filename = filename

    def save_weights(self, model: torch.nn.Module, step: int = 0, overwrite_existing: bool = False) -> None:
        """Saves a PyTorch model to the local filesystem."""
        path = self._get_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite_existing and path.exists():
            path.unlink()

        torch.save(model.state_dict(), path)

    def load_weights(self, model: torch.nn.Module, step: int = 0) -> torch.nn.Module:
        """Loads weights into a PyTorch model from the local filesystem."""
        path = self._get_path(step)
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / self.filename
