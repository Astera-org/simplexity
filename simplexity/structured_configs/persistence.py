"""Persistence configuration dataclasses."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import re
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import _validate_bool, _validate_nonempty_str, _validate_uri
from simplexity.utils.config_utils import dynamic_resolve


@dataclass
class LocalPersisterInstanceConfig(InstanceConfig):
    """Configuration for the local persister."""

    directory: str

    def __init__(self, directory: str, _target_: str = "simplexity.persistence.local_persister.LocalPersister"):
        super().__init__(_target_=_target_)
        self.directory = directory


def is_local_persister_config(cfg: DictConfig, framework: str | None = None) -> bool:
    """Check if the configuration is a LocalPersisterInstanceConfig."""
    if framework is None:
        file_pattern = "local_[a-z]+_persister"
        class_pattern = "Local[A-Z][a-z]+Persister"
    else:
        file_pattern = f"local_{framework.lower()}_persister"
        class_pattern = f"Local{framework.capitalize()}Persister"
    return (
        re.match(f"simplexity.persistence.{file_pattern}.{class_pattern}", OmegaConf.select(cfg, "_target_"))
        is not None
    )


def validate_local_persister_instance_config(cfg: DictConfig, framework: str | None = None) -> None:
    """Validate a LocalPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPersisterInstanceConfig fields (from Hydra).
        framework: The framework of the local persister. If None, the framework will be inferred from the target.
    """
    target = cfg.get("_target_")
    directory = cfg.get("directory")

    validate_instance_config(cfg)
    if not is_local_persister_config(cfg, framework=framework):
        class_name = f"Local{framework.capitalize()}Persister" if framework is not None else "LocalPersister"
        raise ConfigValidationError(f"{class_name}InstanceConfig must be a local persister, got {target}")
    _validate_nonempty_str(directory, "LocalPersisterInstanceConfig.directory")


@dataclass
class LocalEquinoxPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local equinox persister."""

    filename: str = "model.eqx"

    def __init__(
        self,
        directory: str,
        filename: str = "model.eqx",
        _target_: str = "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
    ):
        super().__init__(_target_=_target_, directory=directory)
        self.filename = filename


def is_local_equinox_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalEquinoxPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="equinox")


def validate_local_equinox_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalEquinoxPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalEquinoxPersisterInstanceConfig fields (from Hydra).
    """
    filename = cfg.get("filename")

    validate_local_persister_instance_config(cfg, framework="equinox")
    _validate_nonempty_str(filename, "LocalEquinoxPersisterInstanceConfig.filename")
    assert isinstance(filename, str)
    if not filename.endswith(".eqx"):
        raise ConfigValidationError("LocalEquinoxPersisterInstanceConfig.filename must end with .eqx, got {filename}")


@dataclass
class LocalPenzaiPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local penzai persister."""

    def __init__(
        self, directory: str, _target_: str = "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister"
    ):
        super().__init__(_target_=_target_, directory=directory)


def is_local_penzai_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalPenzaiPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="penzai")


def validate_local_penzai_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalPenzaiPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPenzaiPersisterInstanceConfig fields (from Hydra).
    """
    validate_local_persister_instance_config(cfg, framework="penzai")


@dataclass
class LocalPytorchPersisterInstanceConfig(LocalPersisterInstanceConfig):
    """Configuration for the local pytorch persister."""

    filename: str = "model.pt"

    def __init__(
        self,
        directory: str,
        filename: str = "model.pt",
        _target_: str = "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister",
    ):
        super().__init__(_target_=_target_, directory=directory)
        self.filename = filename


def is_local_pytorch_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a LocalPytorchPersisterInstanceConfig."""
    return is_local_persister_config(cfg, framework="pytorch")


def validate_local_pytorch_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a LocalPytorchPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with LocalPytorchPersisterInstanceConfig fields (from Hydra).
    """
    filename = cfg.get("filename")

    validate_local_persister_instance_config(cfg, framework="pytorch")
    _validate_nonempty_str(filename, "LocalPytorchPersisterInstanceConfig.filename")
    assert isinstance(filename, str)
    if not filename.endswith(".pt"):
        raise ConfigValidationError("LocalPytorchPersisterInstanceConfig.filename must end with .pt, got {filename}")


@dataclass
class MLFlowPersisterInstanceConfig(InstanceConfig):
    """Configuration for the MLflow persister."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True
    artifact_path: str | None = "models"
    config_path: str | None = "config.yaml"

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        artifact_path: str | None = "models",
        config_path: str | None = "config.yaml",
        _target_: str = "simplexity.persistence.mlflow_persister.MLFlowPersister",
    ):
        super().__init__(_target_=_target_)
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.downgrade_unity_catalog = downgrade_unity_catalog
        self.artifact_path = artifact_path
        self.config_path = config_path


def is_mlflow_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a MLFlowPersisterInstanceConfig."""
    return OmegaConf.select(cfg, "_target_") == "simplexity.persistence.mlflow_persister.MLFlowPersister"


def validate_mlflow_persister_instance_config(cfg: DictConfig) -> None:
    """Validate a MLFlowPersisterInstanceConfig.

    Args:
        cfg: A DictConfig with MLFlowPersisterInstanceConfig fields (from Hydra).
    """
    validate_instance_config(cfg, expected_target="simplexity.persistence.mlflow_persister.MLFlowPersister")
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")
    artifact_path = cfg.get("artifact_path")
    config_path = cfg.get("config_path")

    _validate_nonempty_str(experiment_id, "MLFlowPersisterInstanceConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowPersisterInstanceConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowPersisterInstanceConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowPersisterInstanceConfig.run_name", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowPersisterInstanceConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowPersisterInstanceConfig.registry_uri", is_none_allowed=True)
    _validate_bool(
        downgrade_unity_catalog, "MLFlowPersisterInstanceConfig.downgrade_unity_catalog", is_none_allowed=True
    )
    _validate_nonempty_str(artifact_path, "MLFlowPersisterInstanceConfig.artifact_path", is_none_allowed=True)
    _validate_nonempty_str(config_path, "MLFlowPersisterInstanceConfig.config_path", is_none_allowed=True)


@dynamic_resolve
def update_persister_instance_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a PersistenceConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg._target_ = updated_cfg.get("_target_")  # pylint: disable=protected-access
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")
    cfg.artifact_path = updated_cfg.get("artifact_path")
    cfg.config_path = updated_cfg.get("config_path")


@dataclass
class PersistenceConfig:
    """Base configuration for persistence."""

    instance: InstanceConfig
    name: str | None = None


def is_model_persister_target(target: str) -> bool:
    """Check if the target is a model persister target."""
    return target.startswith("simplexity.persistence.")


def is_persister_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a PersistenceInstanceConfig."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_model_persister_target(target)
    return False


def validate_persistence_config(cfg: DictConfig) -> None:
    """Validate a PersistenceConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if not isinstance(instance, DictConfig):
        raise ConfigValidationError("PersistenceConfig.instance is required")
    if is_local_equinox_persister_config(instance):
        validate_local_equinox_persister_instance_config(instance)
    elif is_local_penzai_persister_config(instance):
        validate_local_penzai_persister_instance_config(instance)
    elif is_local_pytorch_persister_config(instance):
        validate_local_pytorch_persister_instance_config(instance)
    elif is_mlflow_persister_config(instance):
        validate_mlflow_persister_instance_config(instance)
    else:
        validate_instance_config(instance)
        if not is_persister_config(instance):
            raise ConfigValidationError("PersistenceConfig.instance must be a persister target")
    _validate_nonempty_str(cfg.get("name"), "PersistenceConfig.name", is_none_allowed=True)
