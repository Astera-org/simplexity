"""Main script for RLVR training using TRL."""

import hydra
import torch
from omegaconf import DictConfig

from simplexity.configs.rlvr_config import RLVRExperimentConfig, validate_rlvr_experiment_config
from simplexity.generative_processes.arithmetic_process import ArithmeticProcess
from simplexity.logging.logger import Logger
from simplexity.training.train_rlvr_model import train_rlvr
from simplexity.utils.hydra import typed_instantiate


@hydra.main(config_path="configs", config_name="train_rlvr_model.yaml", version_base="1.2")
def train_rlvr_model(cfg: RLVRExperimentConfig) -> float:
    """Train a model using RLVR (Reinforcement Learning from Verifier Rewards)."""
    assert isinstance(cfg, DictConfig)
    validate_rlvr_experiment_config(cfg)
    
    # Setup logging
    if cfg.logging:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
    else:
        logger = None
    
    # Setup data generator
    training_data_generator = typed_instantiate(cfg.training_data_generator.instance, ArithmeticProcess)
    
    # Setup model
    model = typed_instantiate(cfg.predictive_model.instance, torch.nn.Module)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Training RLVR model on device: {device}")
    print(f"Model vocabulary size: {training_data_generator.vocab_size}")
    print(f"Training configuration: {cfg.rlvr_training}")
    
    # Train the model
    try:
        trained_model, final_reward = train_rlvr(
            model=model,
            rlvr_cfg=cfg.rlvr_training,
            arithmetic_process=training_data_generator,
            logger=logger,
        )
        
        print(f"Training completed with final reward: {final_reward}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        if logger:
            logger.close()
        raise
    
    # Save final model if persistence is configured
    if cfg.persistence:
        try:
            persister = typed_instantiate(cfg.persistence.instance, type(None))
            if persister and hasattr(persister, 'save_weights'):
                persister.save_weights(trained_model, "final")
                print("Final model saved successfully")
        except Exception as e:
            print(f"Failed to save final model: {e}")
    
    # Close logger
    if logger:
        logger.close()
    
    return final_reward


if __name__ == "__main__":
    train_rlvr_model()