from dataclasses import dataclass

from simplexity.configs.predictive_model.config import Config as ModelConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    predictive_model: ModelConfig

    seed: int
