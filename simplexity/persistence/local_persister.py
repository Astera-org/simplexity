from pathlib import Path

from simplexity.persistence.model_persister import ModelPersister


class LocalPersister(ModelPersister):
    """Persists a model to the local filesystem."""

    directory: Path
