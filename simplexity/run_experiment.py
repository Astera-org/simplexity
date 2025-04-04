import hydra
from omegaconf import DictConfig

from simplexity.configs.config import Config
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train import train
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="experiment.yaml", version_base="1.2")
def run_experiment(cfg: Config) -> float:
    """Run the experiment."""
    assert isinstance(cfg, DictConfig)
    logger = typed_instantiate(cfg.logging.instance, Logger)
    logger.log_config(cfg)
    logger.log_params(cfg)
    generative_process = typed_instantiate(cfg.generative_process.instance, GenerativeProcess)
    initial_gen_process_state = generative_process.initial_state
    vocab_size = generative_process.vocab_size
    model = typed_instantiate(cfg.predictive_model.instance, PredictiveModel, vocab_size=vocab_size)
    persister = typed_instantiate(cfg.persistence.instance, ModelPersister)
    if cfg.predictive_model.load_checkpoint_name:
        model = persister.load_weights(model, cfg.predictive_model.load_checkpoint_name)
    _, loss = train(
        cfg.train,
        model,
        generative_process,
        initial_gen_process_state,
        persister,
        logger,
    )

    logger.close()

    return loss


if __name__ == "__main__":
    run_experiment()
