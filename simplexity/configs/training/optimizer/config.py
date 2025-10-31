from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.configs.instance_config import InstanceConfig


@dataclass
class Config:
    """Base configuration for predictive models."""

    instance: InstanceConfig


def is_pytorch_optimizer_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PyTorch optimizer configuration."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return target.startswith("torch.optim.")
    return False
