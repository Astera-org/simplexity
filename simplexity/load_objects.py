from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from simplexity.configs.load_config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.state_sampler import StateSampler
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.hydra import typed_instantiate


def load_config(config_path: str | Path, config_name: str) -> DictConfig:
    """Load objects from a specific YAML config file."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        return hydra.compose(config_name=config_name)


def load_objects(cfg: Config | DictConfig) -> dict[str, Any]:
    """Load objects from a config."""
    d = {}
    if cfg.generative_process:
        d["generative_process"] = typed_instantiate(cfg.generative_process.instance, GenerativeProcess)
    if cfg.state_sampler:
        d["state_sampler"] = typed_instantiate(cfg.state_sampler.instance, StateSampler)
    if cfg.predictive_model:
        d["model"] = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)
        if cfg.persistence:
            with typed_instantiate(cfg.persistence.instance, ModelPersister) as persister:
                if cfg.predictive_model.load_checkpoint_step:
                    d["model"] = persister.load_weights(d["model"], cfg.predictive_model.load_checkpoint_step)
                    print(f"Loaded model from checkpoint {cfg.predictive_model.load_checkpoint_step}")
    return d
