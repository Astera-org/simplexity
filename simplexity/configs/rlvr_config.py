"""Main configuration for RLVR training experiments."""

from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as DataGeneratorConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig
from simplexity.configs.predictive_model.config import validate_config as validate_model_config
from simplexity.configs.training.rlvr_config import RLVRConfig, validate_rlvr_config


@dataclass
class RLVRExperimentConfig:
    """Configuration for RLVR experiments."""
    
    training_data_generator: DataGeneratorConfig
    predictive_model: ModelConfig
    persistence: PersistenceConfig | None
    logging: LoggingConfig | None
    rlvr_training: RLVRConfig
    
    seed: int
    experiment_name: str
    run_name: str


def validate_rlvr_experiment_config(cfg: RLVRExperimentConfig) -> None:
    """Validate the RLVR experiment configuration."""
    # Validate individual components
    validate_model_config(cfg.predictive_model)
    validate_rlvr_config(cfg.rlvr_training)
    
    # Validate consistency between components
    assert cfg.seed == cfg.rlvr_training.seed, "Seeds must match between experiment and training configs"
    
    # Validate experiment metadata
    assert cfg.experiment_name.strip(), "Experiment name cannot be empty"
    assert cfg.run_name.strip(), "Run name cannot be empty"