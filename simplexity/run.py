import hydra
from omegaconf import DictConfig

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train
from simplexity.utils.hydra import typed_instantiate


def validation_required(cfg: Config) -> bool:
    """Check if validation is required."""
    return (
        cfg.training.validate_every is not None
        and cfg.training.validate_every > 0
        and cfg.training.validate_every <= cfg.training.num_steps
    )


def logging_required(cfg: Config) -> bool:
    """Check if logging is required."""
    if (
        cfg.training.log_every is not None
        and cfg.training.log_every > 0
        and cfg.training.log_every <= cfg.training.num_steps
    ):
        return True
    return bool(
        validation_required(cfg)
        and cfg.validation
        and cfg.validation.log_every is not None
        and cfg.validation.log_every > 0
        and cfg.validation.log_every <= cfg.validation.num_steps
    )


def persistence_required(cfg: Config) -> bool:
    """Check if persistence is required."""
    return cfg.predictive_model.load_checkpoint_step is not None or (
        cfg.training.checkpoint_every is not None
        and cfg.training.checkpoint_every > 0
        and cfg.training.checkpoint_every <= cfg.training.num_steps
    )


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a model."""
    assert isinstance(cfg, DictConfig)

    if cfg.logging:
        assert logging_required(cfg), "Logging is configured but not required"
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        assert not logging_required(cfg), "Logging is required but not configured"
        logger = None

    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)

    if validation_required(cfg):
        assert cfg.validation_data_generator is not None
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        assert cfg.validation_data_generator is None
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    vocab_size = training_data_generator.vocab_size
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, vocab_size=vocab_size)

    if persistence_required(cfg):
        assert cfg.persistence is not None, "Persistence is required but not configured"
        with typed_instantiate(cfg.persistence.instance, ModelPersister) as persister:
            if cfg.predictive_model.load_checkpoint_step:
                model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_step)
            _, loss = train(
                model,
                cfg.training,
                training_data_generator,
                logger,
                cfg.validation,
                validation_data_generator,
                persister,
                training_bos_token=cfg.training_data_generator.bos_token,
                training_eos_token=cfg.training_data_generator.eos_token,
                validation_bos_token=validation_bos_token,
                validation_eos_token=validation_eos_token,
            )
    else:
        assert not cfg.persistence, "Persistence is configured but not required"
        _, loss = train(
            model,
            cfg.training,
            training_data_generator,
            logger,
            cfg.validation,
            validation_data_generator,
            None,
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
