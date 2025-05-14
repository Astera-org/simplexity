from dataclasses import dataclass

from simplexity.configs.generative_process.config import Config as GenerativeProcessConfig
from simplexity.configs.persistence.config import Config as PersistenceConfig
from simplexity.configs.predictive_model.config import Config as PredictiveModelConfig
from simplexity.configs.state_sampler.config import Config as StateSamplerConfig


@dataclass
class Config:
    """Configuration for the experiment."""

    generative_process: GenerativeProcessConfig | None
    state_sampler: StateSamplerConfig | None
    predictive_model: PredictiveModelConfig | None
    persistence: PersistenceConfig | None

    seed: int
