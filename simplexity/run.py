from contextlib import nullcontext
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train
from simplexity.utils.hydra import typed_instantiate


def compute_vocab_and_special_tokens(cfg_generator: Any, generator: GenerativeProcess) -> None:
    """Compute vocab_size and special token IDs based on generator and use_bos/use_eos flags.

    This modifies the config in-place to set:
    - bos_token: generator.vocab_size if use_bos else None
    - eos_token: next available token ID if use_eos else None
    - vocab_size: base vocab + number of special tokens
    """
    base_vocab_size = generator.vocab_size
    num_special_tokens = 0

    if cfg_generator.use_bos:
        OmegaConf.update(cfg_generator, "bos_token", base_vocab_size + num_special_tokens, merge=False)
        num_special_tokens += 1
    else:
        OmegaConf.update(cfg_generator, "bos_token", None, merge=False)

    if cfg_generator.use_eos:
        OmegaConf.update(cfg_generator, "eos_token", base_vocab_size + num_special_tokens, merge=False)
        num_special_tokens += 1
    else:
        OmegaConf.update(cfg_generator, "eos_token", None, merge=False)

    total_vocab_size = base_vocab_size + num_special_tokens
    OmegaConf.update(cfg_generator, "vocab_size", total_vocab_size, merge=False)


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a model."""
    assert isinstance(cfg, DictConfig)
    validate_config(cfg)

    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None

    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)
    compute_vocab_and_special_tokens(cfg.training_data_generator, training_data_generator)

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        compute_vocab_and_special_tokens(cfg.validation_data_generator, validation_data_generator)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)

    persister_context = (
        typed_instantiate(cfg.persistence.instance, ModelPersister) if cfg.persistence else nullcontext()
    )

    with persister_context as persister:
        if isinstance(persister, ModelPersister):
            if cfg.predictive_model.load_checkpoint_step:
                model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_step)
            train_persister = persister
        else:
            train_persister = None

        _, loss = train(
            model,
            cfg.training,
            training_data_generator,
            logger,
            cfg.validation,
            validation_data_generator,
            train_persister,
            training_bos_token=cfg.training_data_generator.bos_token,
            training_eos_token=cfg.training_data_generator.eos_token,
            validation_bos_token=validation_bos_token,
            validation_eos_token=validation_eos_token,
        )

    if logger:
        logger.close()

    return loss


if __name__ == "__main__":
    train_model()
