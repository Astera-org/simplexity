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
        return self.directory / str(step) / self.filename

    # --- Checkpoint discovery ---
    def list_checkpoints(self) -> list[int]:
        steps: list[int] = []
        if not self.directory.exists():
            return steps
        for child in self.directory.iterdir():
            if child.is_dir():
                try:
                    step = int(child.name)
                except ValueError:
                    continue
                if (child / self.filename).exists():
                    steps.append(step)
        steps.sort()
        return steps

    def latest_checkpoint(self) -> int | None:
        steps = self.list_checkpoints()
        return steps[-1] if steps else None

    def checkpoint_exists(self, step: int) -> bool:
        return (self.directory / str(step) / self.filename).exists()

    def uri_for_step(self, step: int) -> str:
        path = self._get_path(step)
        return f"file://{path}"
