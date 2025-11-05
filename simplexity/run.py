from contextlib import nullcontext

import hydra
from omegaconf import DictConfig

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.run_management.structured_configs import MainConfig as Config
from simplexity.run_management.structured_configs import validate_config
from simplexity.training.train_model import train
from simplexity.utils.config_utils import typed_instantiate


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

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
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
