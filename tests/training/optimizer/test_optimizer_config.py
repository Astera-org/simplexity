from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from simplexity.configs.training.optimizer.config import is_pytorch_optimizer_config

CONFIG_DIR = Path("tests/training/optimizer")


def test_is_pytorch_optimizer_config():
    cfg = OmegaConf.load(CONFIG_DIR / "pytorch.yaml")
    assert isinstance(cfg, DictConfig)
    instance_config = cfg.get("instance", None)
    assert instance_config is not None
    assert is_pytorch_optimizer_config(instance_config)


def test_is_not_pytorch_optimizer_config():
    cfg = OmegaConf.load(CONFIG_DIR / "optax.yaml")
    assert isinstance(cfg, DictConfig)
    instance_config = cfg.get("instance", None)
    assert instance_config is not None
    assert not is_pytorch_optimizer_config(instance_config)
