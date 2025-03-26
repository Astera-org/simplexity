from dataclasses import dataclass

from simplexity.configs.train.optimizer.config import Config as OptimizerConfig


@dataclass
class Config:
    """Configuration for the training process."""

    seed: int
    sequence_len: int
    batch_size: int
    num_steps: int
    log_every: int
    validate_every: int
    num_validation_steps: int
    checkpoint_every: int
    checkpoint_name: str
    optimizer: OptimizerConfig
