from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as ProcessConfig
from simplexity.configs.predictive_model.config import Config as ModelConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    generative_process: ProcessConfig
    predictive_model: ModelConfig

    seed: int
