from dataclasses import dataclass
from typing import Literal


@dataclass
class LoggingInstanceConfig:
    """Configuration for the logging instance."""

    _target_: Literal[
        "simplexity.logging.file_logger.FileLogger",
        "simplexity.logging.mlflow_logger.MLFlowLogger",
        "simplexity.logging.print_logger.PrintLogger",
    ]


@dataclass
class FileLoggerConfig(LoggingInstanceConfig):
    """Configuration for file logger."""

    # _target_: FileLogger
    file_path: str


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

    name: Literal["file_logger", "mlflow_logger", "print_logger"]
    instance: LoggingInstanceConfig
