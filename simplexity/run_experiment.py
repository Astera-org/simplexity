import hydra
from omegaconf import DictConfig

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.state_sampler import StateSampler
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_equinox_model import train
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config) -> float:
    """Run the experiment."""
    assert isinstance(cfg, DictConfig)
    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)
    if cfg.training_state_sampler:
        training_state_sampler = typed_instantiate(cfg.training_state_sampler.instance, StateSampler)
    else:
        training_state_sampler = None
    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None
    if cfg.validation_state_sampler:
        validation_state_sampler = typed_instantiate(cfg.validation_state_sampler.instance, StateSampler)
    else:
        validation_state_sampler = None
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel)
    if cfg.persistence:
        with typed_instantiate(cfg.persistence.instance, ModelPersister) as persister:
            if cfg.predictive_model.load_checkpoint_step:
                model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_step)
            model, loss = train(
                model,
                cfg.training,
                training_data_generator,
                training_state_sampler,
                logger,
                cfg.validation,
                validation_data_generator,
                validation_state_sampler,
                persister,
                training_bos_token=cfg.training_data_generator.bos_token,
                training_eos_token=cfg.training_data_generator.eos_token,
                validation_bos_token=validation_bos_token,
                validation_eos_token=validation_eos_token,
            )
            persister.save_weights(model, cfg.training.num_steps, overwrite_existing=True)
    else:
        _, loss = train(
            model,
            cfg.training,
            training_data_generator,
            training_state_sampler,
            logger,
            cfg.validation,
            validation_data_generator,
            validation_state_sampler,
            training_bos_token=cfg.training_data_generator.bos_token,
            training_eos_token=cfg.training_data_generator.eos_token,
            validation_bos_token=validation_bos_token,
            validation_eos_token=validation_eos_token,
        )
    if logger:
        logger.close()

    return loss


if __name__ == "__main__":
    run_experiment()
