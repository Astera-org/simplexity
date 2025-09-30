import hydra
import torch
from omegaconf import DictConfig

from simplexity.configs.config import Config, validate_config
from simplexity.generative_processes.arithmetic_process import ArithmeticProcess
from simplexity.logging.logger import Logger
from simplexity.training.train_pytorch_model import train
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="addition_pretraining.yaml", version_base="1.2")
def train_model(cfg: Config) -> float:
    """Train a model."""
    assert isinstance(cfg, DictConfig)
    validate_config(cfg)

    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_git_info()
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None

    # Use ArithmeticProcess for arithmetic data generators
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, ArithmeticProcess)

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, ArithmeticProcess)
        validation_bos_token = cfg.validation_data_generator.bos_token
        validation_eos_token = cfg.validation_data_generator.eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    model = typed_instantiate(cfg.predictive_model.instance, torch.nn.Module)

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
