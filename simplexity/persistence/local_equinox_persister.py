from pathlib import Path

import equinox as eqx

from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.predictive_model import PredictiveModel


class LocalEquinoxPersister(LocalPersister):
    """Persists a model to the local filesystem."""

    filename: str = "model.eqx"

    def __init__(self, directory: str | Path, filename: str = "model.eqx"):
        self.directory = Path(directory)
        self.filename = filename

    def save_weights(self, model: PredictiveModel, step: int = 0, overwrite_existing: bool = False) -> None:
        """Saves a model to the local filesystem."""
        path = self._get_path(step)
        if path.exists():
            if overwrite_existing:
                path.unlink()
            else:
                raise FileExistsError(f"File {path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, model)

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        path = self._get_path(step)
        return eqx.tree_deserialise_leaves(path, model)

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / self.filename
