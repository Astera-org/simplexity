import hydra
from omegaconf import DictConfig

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train import train


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config) -> float:
    """Run the experiment."""
    assert isinstance(cfg, DictConfig)
    logger = typed_instantiate(cfg.logging.instance, Logger)
    logger.log_config(cfg)
    logger.log_params(cfg)
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)
    validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
    vocab_size = training_data_generator.vocab_size
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, vocab_size=vocab_size)
    persister = typed_instantiate(cfg.persistence.instance, ModelPersister)
    if cfg.predictive_model.load_checkpoint_name:
        model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_name)
    _, loss = train(
        model,
        cfg.training,
        training_data_generator,
        logger,
        cfg.validation,
        validation_data_generator,
        persister,
    )

    logger.close()

    return loss


if __name__ == "__main__":
    run_experiment()
