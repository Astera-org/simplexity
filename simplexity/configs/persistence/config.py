from dataclasses import dataclass
from typing import Literal


@dataclass
class PersistenceInstanceConfig:
    """Configuration for the persistence instance."""

    _target_: Literal["simplexity.persistence.local_persister.LocalPersister"]


@dataclass
class LocalPersisterConfig(PersistenceInstanceConfig):
    """Configuration for local persister."""

    _target_: Literal["simplexity.persistence.local_persister.LocalPersister"]
    base_dir: str


@dataclass
class Config:
    """Base configuration for persistence."""

    name: Literal["local_persister"]
    instance: PersistenceInstanceConfig
