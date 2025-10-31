from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.configs.instance_config import InstanceConfig


@dataclass
class Config:
    """Base configuration for persistence."""

    name: str
    instance: InstanceConfig


def is_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PersistenceInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.persistence.")
    return False
