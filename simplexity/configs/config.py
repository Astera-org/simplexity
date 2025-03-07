from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the experiment."""

    seed: int
