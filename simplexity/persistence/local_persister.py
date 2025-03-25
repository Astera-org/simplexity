from pathlib import Path

import equinox as eqx

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel


class LocalPersister(ModelPersister):
    """Persists a model to the local filesystem."""

    base_dir: str

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def save_weights(self, model: PredictiveModel, name: str) -> None:
        """Saves a model to the local filesystem."""
        path = self._get_path(name)
        eqx.tree_serialise_leaves(path, model)

    def load_weights(self, model: PredictiveModel, name: str) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        path = self._get_path(name)
        return eqx.tree_deserialise_leaves(path, model)

    def _get_path(self, name: str) -> Path:
        path = Path(self.base_dir) / name
        if not path.suffix:
            path = path.with_suffix(".eqx")
        return path
