from contextlib import nullcontext

import hydra
from omegaconf import DictConfig

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
    assert isinstance(cfg, DictConfig)
    validate_config(cfg)

    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None

    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, GenerativeProcess)
    
    # Compute training token values and model vocab size
    base_vocab_size = training_data_generator.vocab_size
    use_bos = getattr(cfg.training_data_generator, 'use_bos_token', False)
    use_eos = getattr(cfg.training_data_generator, 'use_eos_token', False)
    
    if use_bos and use_eos:
        training_bos_token = base_vocab_size
        training_eos_token = base_vocab_size + 1
        model_vocab_size = base_vocab_size + 2
    elif use_bos:
        training_bos_token = base_vocab_size
        training_eos_token = None
        model_vocab_size = base_vocab_size + 1
    elif use_eos:
        training_bos_token = None
        training_eos_token = base_vocab_size
        model_vocab_size = base_vocab_size + 1
    else:
        training_bos_token = None
        training_eos_token = None
        model_vocab_size = base_vocab_size
    
    # Add computed values to config for logging
    cfg.training_data_generator.computed_vocab_size = base_vocab_size
    cfg.training_data_generator.computed_model_vocab_size = model_vocab_size
    if training_bos_token is not None:
        cfg.training_data_generator.computed_bos_token = training_bos_token
    if training_eos_token is not None:
        cfg.training_data_generator.computed_eos_token = training_eos_token

    if cfg.validation_data_generator:
        validation_data_generator = typed_instantiate(cfg.validation_data_generator.instance, GenerativeProcess)
        
        # Compute validation token values
        val_base_vocab_size = validation_data_generator.vocab_size
        val_use_bos = getattr(cfg.validation_data_generator, 'use_bos_token', False)
        val_use_eos = getattr(cfg.validation_data_generator, 'use_eos_token', False)
        
        if val_use_bos and val_use_eos:
            validation_bos_token = val_base_vocab_size
            validation_eos_token = val_base_vocab_size + 1
            val_model_vocab_size = val_base_vocab_size + 2
        elif val_use_bos:
            validation_bos_token = val_base_vocab_size
            validation_eos_token = None
            val_model_vocab_size = val_base_vocab_size + 1
        elif val_use_eos:
            validation_bos_token = None
            validation_eos_token = val_base_vocab_size
            val_model_vocab_size = val_base_vocab_size + 1
        else:
            validation_bos_token = None
            validation_eos_token = None
            val_model_vocab_size = val_base_vocab_size
            
        # Add computed values to config for logging
        cfg.validation_data_generator.computed_vocab_size = val_base_vocab_size
        cfg.validation_data_generator.computed_model_vocab_size = val_model_vocab_size
        if validation_bos_token is not None:
            cfg.validation_data_generator.computed_bos_token = validation_bos_token
        if validation_eos_token is not None:
            cfg.validation_data_generator.computed_eos_token = validation_eos_token
    else:
        validation_data_generator = None
        validation_bos_token = None
        validation_eos_token = None

    # Override the vocab_size in model config with computed model_vocab_size
    if hasattr(cfg.predictive_model.instance, 'vocab_size'):
        cfg.predictive_model.instance.vocab_size = model_vocab_size
    elif hasattr(cfg.predictive_model.instance, 'config') and hasattr(cfg.predictive_model.instance.config, 'vocab_size'):
        cfg.predictive_model.instance.config.vocab_size = model_vocab_size
    
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
            training_bos_token=training_bos_token,
            training_eos_token=training_eos_token,
            validation_bos_token=validation_bos_token,
            validation_eos_token=validation_eos_token,
        )

    if logger:
        logger.close()

    return loss


if __name__ == "__main__":
    train_model()
