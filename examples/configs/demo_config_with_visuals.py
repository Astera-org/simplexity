"""Configuration schema for the activation visualization demo."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass, field

from simplexity.run_management.structured_configs import (
    ActivationTrackerConfig,
    GenerativeProcessConfig,
    LoggingConfig,
    MLFlowConfig,
    OptimizerConfig,
    PersistenceConfig,
    PredictiveModelConfig,
)


@dataclass
class Config:
    """Structured config used by the visualization demo."""

    mlflow: MLFlowConfig | None = None
    logging: LoggingConfig | None = None
    generative_process: GenerativeProcessConfig | None = None
    persistence: PersistenceConfig | None = None
    predictive_model: PredictiveModelConfig | None = None
    optimizer: OptimizerConfig | None = None
    activation_tracker: ActivationTrackerConfig | None = None
    experiment_name: str = "activation_tracker_demo"
    run_name: str = "activation_tracker_demo"
    seed: int = 0
    tags: dict[str, str] = field(default_factory=dict)
    training_steps: int = 5
    validate_every: int = 1
    visualization_dir: str = "activation_visualizations"
