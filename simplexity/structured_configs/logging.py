"""Logging configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, _validate_instance_config
from simplexity.structured_configs.validation import _validate_bool, _validate_nonempty_str, _validate_uri
from simplexity.utils.config_utils import dynamic_resolve


@dataclass
class FileLoggerInstanceConfig(InstanceConfig):
    """Configuration for FileLogger."""

    file_path: str

    def __init__(self, file_path: str, _target_: str = "simplexity.logging.file_logger.FileLogger") -> None:
        super().__init__(_target_=_target_)
        self.file_path = file_path


def is_file_logger_target(target: str) -> bool:
    """Check if the target is a file logger target."""
    return target == "simplexity.logging.file_logger.FileLogger"


def is_file_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a FileLoggerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_file_logger_target(target)
    return False


def validate_file_logger_instance_config(cfg: DictConfig) -> None:
    """Validate a FileLoggerInstanceConfig.

    Args:
        cfg: A DictConfig with FileLoggerInstanceConfig fields (from Hydra).
    """
    file_path = cfg.get("file_path")

    _validate_instance_config(cfg, expected_target="simplexity.logging.file_logger.FileLogger")
    _validate_nonempty_str(file_path, "FileLoggerInstanceConfig.file_path")


@dataclass
class MLFlowLoggerInstanceConfig(InstanceConfig):
    """Configuration for MLFlowLogger."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True


def is_mlflow_logger_target(target: str) -> bool:
    """Check if the target is a mlflow logger target."""
    return target == "simplexity.logging.mlflow_logger.MLFlowLogger"


def is_mlflow_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a MLFlowLoggerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_mlflow_logger_target(target)
    return False


def validate_mlflow_logger_instance_config(cfg: DictConfig) -> None:
    """Validate a MLFlowLoggerInstanceConfig.

    Args:
        cfg: A DictConfig with MLFlowLoggerInstanceConfig fields (from Hydra).
    """
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")

    _validate_instance_config(cfg, expected_target="simplexity.logging.mlflow_logger.MLFlowLogger")
    _validate_nonempty_str(experiment_id, "MLFlowLoggerInstanceConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowLoggerInstanceConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowLoggerInstanceConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowLoggerInstanceConfig.run_name", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowLoggerInstanceConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowLoggerInstanceConfig.registry_uri", is_none_allowed=True)
    _validate_bool(downgrade_unity_catalog, "MLFlowLoggerInstanceConfig.downgrade_unity_catalog", is_none_allowed=True)


@dynamic_resolve
def update_logging_instance_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a LoggingInstanceConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg._target_ = updated_cfg.get("_target_")  # pylint: disable=protected-access
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")


@dataclass
class LoggingConfig:
    """Base configuration for logging."""

    instance: InstanceConfig
    name: str | None = None


def is_logger_target(target: str) -> bool:
    """Check if the target is a logger target."""
    return target.startswith("simplexity.logging.")


def is_logger_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LoggingInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_logger_target(target)
    return False


def validate_logging_config(cfg: DictConfig) -> None:
    """Validate a LoggingConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    name = cfg.get("name")

    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("LoggingConfig.instance must be a DictConfig")
    if not is_logger_config(instance):
        raise ConfigValidationError("LoggingConfig.instance must be a logger target")

    if is_file_logger_config(instance):
        validate_file_logger_instance_config(instance)
    elif is_mlflow_logger_config(instance):
        validate_mlflow_logger_instance_config(instance)
    else:
        _validate_instance_config(instance)
    _validate_nonempty_str(name, "LoggingConfig.name", is_none_allowed=True)

