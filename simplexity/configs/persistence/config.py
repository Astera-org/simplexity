from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig


@dataclass
class PersistenceInstanceConfig(DictConfig):
    """Configuration for the persistence instance."""

    _target_: Literal[
        "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
        "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister",
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
class S3PersisterConfig(PersistenceInstanceConfig):
    """Configuration for S3 persister."""

    filename: str
    model_framework: str


@dataclass
class Config(DictConfig):
    """Base configuration for persistence."""

    name: Literal["local_equinox_persister", "local_penzai_persister", "s3_persister"]
    instance: PersistenceInstanceConfig
