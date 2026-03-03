"""Tracking utilities."""

from pathlib import Path

from simplexity.predictive_models.types import ModelFramework
from simplexity.tracking.model_persistence.local_model_persister import (
    LocalModelPersister,
)


def build_local_persister(model_framework: ModelFramework, artifact_dir: Path) -> LocalModelPersister:
    """Build a local persister."""
    if model_framework == ModelFramework.EQUINOX:
        from simplexity.tracking.model_persistence.local_equinox_persister import (  # pylint: disable=import-outside-toplevel
            LocalEquinoxPersister,
        )

        directory = artifact_dir / "equinox"
        return LocalEquinoxPersister(directory=directory)
    if model_framework == ModelFramework.PENZAI:
        from simplexity.tracking.model_persistence.local_penzai_persister import (  # pylint: disable=import-outside-toplevel
            LocalPenzaiPersister,
        )

        directory = artifact_dir / "penzai"
        return LocalPenzaiPersister(directory=directory)
    if model_framework == ModelFramework.PYTORCH:
        from simplexity.tracking.model_persistence.local_pytorch_persister import (  # pylint: disable=import-outside-toplevel
            LocalPytorchPersister,
        )

        directory = artifact_dir / "pytorch"
        return LocalPytorchPersister(directory=directory)

    raise ValueError(f"Unsupported model framework: {model_framework}")
