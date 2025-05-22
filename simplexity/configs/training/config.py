from dataclasses import dataclass

from simplexity.configs.training.optimizer.config import Config as OptimizerConfig


@dataclass
class Config:
    """Configuration for the training process."""

    seed: int
    sequence_len: int
    batch_size: int
    num_steps: int
    log_every: int | None
    validate_every: int | None
    checkpoint_every: int | None
    checkpoint_name: str | None
    optimizer: OptimizerConfig
