from dataclasses import dataclass
from typing import Literal


@dataclass
class PersistenceInstanceConfig:
    """Configuration for the persistence instance."""

    _target_: Literal[
        "simplexity.persistence.local_persister.LocalPersister",
        "simplexity.persistence.s3_persister.S3Persister.from_client_args",
    ]


@dataclass
class LocalPersisterConfig(PersistenceInstanceConfig):
    """Configuration for local persister."""

    base_dir: str


@dataclass
class S3PersisterConfig(PersistenceInstanceConfig):
    """Configuration for S3 persister."""

    config_file: str


@dataclass
class Config:
    """Base configuration for persistence."""

    name: Literal["local_persister", "s3_persister"]
    instance: PersistenceInstanceConfig
