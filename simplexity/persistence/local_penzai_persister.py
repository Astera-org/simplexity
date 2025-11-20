"""Local Penzai persister."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import orbax.checkpoint as ocp
from orbax.checkpoint.handlers import DefaultCheckpointHandlerRegistry
from penzai import pz
from penzai.nn.layer import Layer as PenzaiModel

from simplexity.persistence.local_persister import LocalPersister
from simplexity.utils.penzai_utils import deconstruct_variables, reconstruct_variables


class LocalPenzaiPersister(LocalPersister):
    """Persists a model to the local filesystem."""

    registry: DefaultCheckpointHandlerRegistry

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        self.registry.add("default", ocp.args.PyTreeSave, ocp.PyTreeCheckpointHandler)
        self.registry.add("default", ocp.args.PyTreeRestore, ocp.PyTreeCheckpointHandler)

    def save_weights(self, model: PenzaiModel, step: int = 0, overwrite_existing: bool = False) -> None:
        """Saves a model to the local filesystem."""
        _, variable_values = pz.unbind_variables(model, freeze=True)
        items = deconstruct_variables(variable_values)
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        if overwrite_existing and step in mngr.all_steps():
            mngr.delete(step)
        mngr.save(step=step, args=ocp.args.PyTreeSave(item=items))  # pyright: ignore
        mngr.wait_until_finished()

    def load_weights(self, model: PenzaiModel, step: int = 0) -> PenzaiModel:
        """Loads a model from the local filesystem."""
        mngr = ocp.CheckpointManager(self.directory, handler_registry=self.registry)
        items = mngr.restore(step=step)
        unbound_model, _ = pz.unbind_variables(model)
        variable_values = reconstruct_variables(items)
        return pz.bind_variables(unbound_model, variable_values)
