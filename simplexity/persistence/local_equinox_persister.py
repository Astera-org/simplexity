"""Local Equinox persister."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import equinox as eqx

from simplexity.persistence.local_persister import LocalPersister


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
