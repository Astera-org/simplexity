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
        if overwrite_existing and step in mngr.all_steps():
            mngr.delete(step)
        mngr.save(step=step, args=ocp.args.PyTreeSave(item=items))  # pyright: ignore
        mngr.wait_until_finished()

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Loads a model from the local filesystem."""
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        items = mngr.restore(step=step)
        unbound_model, _ = pz.unbind_variables(model)
        variable_values = reconstruct_variables(items)
        return pz.bind_variables(unbound_model, variable_values)

    # --- Checkpoint discovery ---
    def list_checkpoints(self) -> list[int]:
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        steps = list(mngr.all_steps())
        steps.sort()
        return steps

    def latest_checkpoint(self) -> int | None:
        steps = self.list_checkpoints()
        return steps[-1] if steps else None

    def checkpoint_exists(self, step: int) -> bool:
        return step in set(self.list_checkpoints())

    def uri_for_step(self, step: int) -> str:
        # Orbax uses a directory per checkpoint step
        path = self.directory / str(step)
        return f"file://{path}"
