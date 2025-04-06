from pathlib import Path

import orbax.checkpoint as ocp
from penzai import pz

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.penzai import deconstruct_variables, reconstruct_variables


class LocalPenzaiPersister(ModelPersister):
    """Persists a model to the local filesystem."""

    directory: Path
    mngr: ocp.CheckpointManager

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.mngr = ocp.CheckpointManager(directory, checkpointers=ocp.PyTreeCheckpointer())

    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Saves a model to the local filesystem."""
        _, variable_values = pz.unbind_variables(model, freeze=True)
        items = deconstruct_variables(variable_values)
        self.mngr.save(step=step, items=items)
        self.mngr.wait_until_finished()

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        items = self.mngr.restore(step=step)
        variable_values = reconstruct_variables(items)
        return pz.bind_variables(model, variable_values)
