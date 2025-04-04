from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the validation."""

    seed: int
    sequence_len: int
    batch_size: int
    num_steps: int
    log_every: int
