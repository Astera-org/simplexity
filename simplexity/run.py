import hydra

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a model."""
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

    vocab_size = training_data_generator.vocab_size
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, vocab_size=vocab_size)

    if cfg.persistence:
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
