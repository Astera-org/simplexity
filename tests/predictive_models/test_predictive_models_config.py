from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from simplexity.run_management.structured_configs import is_hooked_transformer_config

CONFIG_DIR = Path("tests/predictive_models")


def test_is_hooked_transformer_config():
    cfg = OmegaConf.load(CONFIG_DIR / "hooked_transformer.yaml")
    assert isinstance(cfg, DictConfig)
    instance_config = cfg.get("instance", None)
    assert instance_config is not None
    assert is_hooked_transformer_config(instance_config)


def test_is_not_hooked_transformer_config():
    cfg = OmegaConf.load(CONFIG_DIR / "gru_rnn.yaml")
    assert isinstance(cfg, DictConfig)
    instance_config = cfg.get("instance", None)
    assert instance_config is not None
    assert not is_hooked_transformer_config(instance_config)
