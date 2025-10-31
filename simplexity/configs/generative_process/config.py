from dataclasses import dataclass

from omegaconf import MISSING, DictConfig

from simplexity.configs.instance_config import InstanceConfig


@dataclass
class Config:
    """Base configuration for predictive models."""

    name: str
    vocab_size: int  # TODO: Make missing
    instance: InstanceConfig
    bos_token: int | None = MISSING
    eos_token: int | None = MISSING


def is_generative_process_config(cfg: DictConfig) -> bool:
    """Check if the configuration is ."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("simplexity.persistence.")
    return False
