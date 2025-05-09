from pathlib import Path

import orbax.checkpoint as ocp
from orbax.checkpoint.handlers import DefaultCheckpointHandlerRegistry
from penzai import pz

from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.penzai import deconstruct_variables, reconstruct_variables


class LocalPenzaiPersister(LocalPersister):
    """Persists a model to the local filesystem."""

    registry: DefaultCheckpointHandlerRegistry

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        self.registry.add("default", ocp.args.PyTreeSave, ocp.PyTreeCheckpointHandler)
        self.registry.add("default", ocp.args.PyTreeRestore, ocp.PyTreeCheckpointHandler)

    def save_weights(self, model: PredictiveModel, step: int = 0, overwrite_existing: bool = False) -> None:
        """Saves a model to the local filesystem."""
        _, variable_values = pz.unbind_variables(model, freeze=True)
        items = deconstruct_variables(variable_values)
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        if step in mngr.all_steps():
            if overwrite_existing:
                mngr.delete(step)
            else:
                raise FileExistsError(f"Checkpoint {step} already exists.")
        mngr.save(step=step, args=ocp.args.PyTreeSave(item=items))  # pyright: ignore
        mngr.wait_until_finished()

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        items = mngr.restore(step=step)
        unbound_model, _ = pz.unbind_variables(model)
        variable_values = reconstruct_variables(items)
        return pz.bind_variables(unbound_model, variable_values)
