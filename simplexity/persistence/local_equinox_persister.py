from pathlib import Path

import equinox as eqx

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel


class LocalEquinoxPersister(ModelPersister):
    """Persists a model to the local filesystem."""

    directory: Path

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Saves a model to the local filesystem."""
        path = self._get_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, model)

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        path = self._get_path(step)
        return eqx.tree_deserialise_leaves(path, model)

    def _get_path(self, step: int) -> Path:
        return self.directory / str(step) / "model.eqx"
