from dataclasses import dataclass
from typing import Literal


@dataclass
class PersistenceInstanceConfig:
    """Configuration for the persistence instance."""

    _target_: Literal[
        "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
        "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister",
        "simplexity.persistence.mlflow_persister.MLFlowPersister",
        "simplexity.persistence.s3_persister.S3Persister.from_config",
    ]


@dataclass
class LocalEquinoxPersisterConfig(PersistenceInstanceConfig):
    """Configuration for local equinox persister."""

    directory: str
    filename: str


@dataclass
class LocalPenzaiPersisterConfig(PersistenceInstanceConfig):
    """Configuration for local penzai persister."""

    directory: str


@dataclass
class MLFlowPersisterConfig(PersistenceInstanceConfig):
    """Configuration for MLflow persister."""

    experiment_name: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    artifact_path: str = "models"
    model_framework: str = "pytorch"
    registered_model_name: str | None = None
    downgrade_unity_catalog: bool = True


@dataclass
class S3PersisterConfig(PersistenceInstanceConfig):
    """Configuration for S3 persister."""

    prefix: str
    model_framework: str
    config_filename: str = "config.ini"


@dataclass
class Config:
    """Base configuration for persistence."""

    name: Literal["local_equinox_persister", "local_penzai_persister", "mlflow_persister", "s3_persister"]
    instance: PersistenceInstanceConfig
