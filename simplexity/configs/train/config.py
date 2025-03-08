from dataclasses import dataclass

from simplexity.configs.train.optimizer.config import Config as OptimizerConfig


@dataclass
class Config:
    """Configuration for the training process."""

    sequence_len: int
    batch_size: int
    num_epochs: int
    optimizer: OptimizerConfig
