from dataclasses import dataclass
from typing import Literal


@dataclass
class LoggingInstanceConfig:
    """Configuration for the logging instance."""

    _target_: Literal["simplexity.logging.mlflow_logger.MLFlowLogger", "simplexity.logging.print_logger.PrintLogger"]


@dataclass
class MLFlowLoggerConfig(LoggingInstanceConfig):
    """Configuration for MLFlow logger."""

    # _target_: MLFlowLogger
    experiment_name: str
    run_name: str
    tracking_uri: str


@dataclass
class PrintLoggerConfig(LoggingInstanceConfig):
    """Configuration for print logger."""

    # _target_: PrintLogger


@dataclass
class Config:
    """Base configuration for logging."""

    name: Literal["mlflow_logger", "print_logger"]
    instance: LoggingInstanceConfig
