from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.configs.instance_config import InstanceConfig


@dataclass
class Config:
    """Base configuration for logging."""

    name: str
    instance: InstanceConfig


def is_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LoggingInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.logging.")
    return False
