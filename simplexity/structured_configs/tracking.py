"""Tracking configuration dataclasses."""

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
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import validate_bool, validate_nonempty_str, validate_uri
from simplexity.utils.config_utils import dynamic_resolve


@dataclass
class FileTrackerInstanceConfig(InstanceConfig):
    """Configuration for FileTracker."""

    file_path: str
    model_dir_name: str = "models"

    def __init__(
        self,
        file_path: str,
        model_dir_name: str = "models",
        _target_: str = "simplexity.tracking.file_tracker.FileTracker",
    ) -> None:
        super().__init__(_target_=_target_)
        self.file_path = file_path
        self.model_dir_name = model_dir_name


def is_file_tracker_target(target: str) -> bool:
    """Check if the target is a file tracker target."""
    return target == "simplexity.tracking.file_tracker.FileTracker"


def is_file_tracker_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a FileTrackerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_file_tracker_target(target)
    return False


def validate_file_tracker_instance_config(cfg: DictConfig) -> None:
    """Validate a FileTrackerInstanceConfig."""
    file_path = cfg.get("file_path")
    model_dir_name = cfg.get("model_dir_name")

    validate_instance_config(cfg, expected_target="simplexity.tracking.file_tracker.FileTracker")
    validate_nonempty_str(file_path, "FileTrackerInstanceConfig.file_path")
    validate_nonempty_str(model_dir_name, "FileTrackerInstanceConfig.model_dir_name", is_none_allowed=True)


@dataclass
class MlflowTrackerInstanceConfig(InstanceConfig):
    """Configuration for MlflowTracker."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True
    model_dir: str = "models"
    config_path: str = "config.yaml"

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        model_dir: str = "models",
        config_path: str = "config.yaml",
        _target_: str = "simplexity.tracking.mlflow_tracker.MlflowTracker",
    ) -> None:
        super().__init__(_target_=_target_)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.downgrade_unity_catalog = downgrade_unity_catalog
        self.model_dir = model_dir
        self.config_path = config_path


def is_mlflow_tracker_target(target: str) -> bool:
    """Check if the target is a mlflow tracker target."""
    return target == "simplexity.tracking.mlflow_tracker.MlflowTracker"


def is_mlflow_tracker_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a MlflowTrackerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_mlflow_tracker_target(target)
    return False


def validate_mlflow_tracker_instance_config(cfg: DictConfig) -> None:
    """Validate a MlflowTrackerInstanceConfig."""
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")
    model_dir = cfg.get("model_dir")
    config_path = cfg.get("config_path")

    validate_instance_config(cfg, expected_target="simplexity.tracking.mlflow_tracker.MlflowTracker")
    validate_nonempty_str(experiment_id, "MlflowTrackerInstanceConfig.experiment_id", is_none_allowed=True)
    validate_nonempty_str(experiment_name, "MlflowTrackerInstanceConfig.experiment_name", is_none_allowed=True)
    validate_nonempty_str(run_id, "MlflowTrackerInstanceConfig.run_id", is_none_allowed=True)
    validate_nonempty_str(run_name, "MlflowTrackerInstanceConfig.run_name", is_none_allowed=True)
    validate_uri(tracking_uri, "MlflowTrackerInstanceConfig.tracking_uri", is_none_allowed=True)
    validate_uri(registry_uri, "MlflowTrackerInstanceConfig.registry_uri", is_none_allowed=True)
    validate_bool(downgrade_unity_catalog, "MlflowTrackerInstanceConfig.downgrade_unity_catalog", is_none_allowed=True)
    validate_nonempty_str(model_dir, "MlflowTrackerInstanceConfig.model_dir", is_none_allowed=True)
    validate_nonempty_str(config_path, "MlflowTrackerInstanceConfig.config_path", is_none_allowed=True)


@dataclass
class S3TrackerInstanceConfig(InstanceConfig):
    """Configuration for S3Tracker (from_config factory)."""

    prefix: str
    config_filename: str = "config.ini"

    def __init__(
        self,
        prefix: str,
        config_filename: str = "config.ini",
        _target_: str = "simplexity.tracking.s3_tracker.S3Tracker.from_config",
    ) -> None:
        super().__init__(_target_=_target_)
        self.prefix = prefix
        self.config_filename = config_filename


def is_s3_tracker_target(target: str) -> bool:
    """Check if the target is a s3 tracker target."""
    return target == "simplexity.tracking.s3_tracker.S3Tracker.from_config"


def is_s3_tracker_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a S3TrackerInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_s3_tracker_target(target)
    return False


def validate_s3_tracker_instance_config(cfg: DictConfig) -> None:
    """Validate a S3TrackerInstanceConfig."""
    prefix = cfg.get("prefix")
    config_filename = cfg.get("config_filename")

    validate_instance_config(cfg, expected_target="simplexity.tracking.s3_tracker.S3Tracker.from_config")
    validate_nonempty_str(prefix, "S3TrackerInstanceConfig.prefix")
    validate_nonempty_str(config_filename, "S3TrackerInstanceConfig.config_filename")


@dynamic_resolve
def update_tracking_instance_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a TrackingInstanceConfig with the updated configuration."""
    cfg.merge_with(updated_cfg)


@dataclass
class TrackingConfig:
    """Base configuration for tracking."""

    instance: InstanceConfig
    name: str | None = None


def is_run_tracker_target(target: str) -> bool:
    """Check if the target is a run tracker target."""
    return target.startswith("simplexity.tracking.")


def is_run_tracker_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a TrackingInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_run_tracker_target(target)
    return False


def validate_tracking_config(cfg: DictConfig) -> None:
    """Validate a TrackingConfig."""
    instance = cfg.get("instance")
    name = cfg.get("name")

    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("TrackingConfig.instance must be a DictConfig")

    if is_file_tracker_config(instance):
        validate_file_tracker_instance_config(instance)
    elif is_mlflow_tracker_config(instance):
        validate_mlflow_tracker_instance_config(instance)
    elif is_s3_tracker_config(instance):
        validate_s3_tracker_instance_config(instance)
    else:
        validate_instance_config(instance)
        if not is_run_tracker_config(instance):
            raise ConfigValidationError("TrackingConfig.instance must be a tracker target")
    validate_nonempty_str(name, "TrackingConfig.name", is_none_allowed=True)
