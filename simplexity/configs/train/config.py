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
    optimizer: OptimizerConfig
    checkpoint_every: int
    checkpoint_name: str
