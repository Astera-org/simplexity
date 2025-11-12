"""Local Equinox persister."""

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx

from simplexity.persistence.local_persister import LocalPersister, LocalPersisterInstanceConfig


@dataclass
class LocalEquinoxPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local equinox persister."""

    filename: str = "model.eqx"


class LocalEquinoxPersister(LocalPersister):
    """Persists a model to the local filesystem."""

    filename: str = "model.eqx"

    def __init__(self, directory: str | Path, filename: str = "model.eqx"):
        self.directory = Path(directory)
        self.filename = filename

    def save_weights(self, model: eqx.Module, step: int = 0) -> None:
        """Saves a model to the local filesystem."""
        path = self._get_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, model)

    def load_weights(self, model: eqx.Module, step: int = 0) -> eqx.Module:
        """Loads a model from the local filesystem."""
        path = self._get_path(step)
        return eqx.tree_deserialise_leaves(path, model)

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / self.filename
