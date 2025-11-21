"""MLflow configuration dataclasses."""

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

from simplexity.structured_configs.validation import _validate_bool, _validate_nonempty_str, _validate_uri
from simplexity.utils.config_utils import dynamic_resolve


@dataclass
class MLFlowConfig:
    """Configuration for MLflow."""

    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool | None = None


def validate_mlflow_config(cfg: DictConfig) -> None:
    """Validate an MLFlowConfig.

    Args:
        cfg: A DictConfig with MLFlowConfig fields (from Hydra).
    """
    experiment_id = cfg.get("experiment_id")
    experiment_name = cfg.get("experiment_name")
    run_id = cfg.get("run_id")
    run_name = cfg.get("run_name")
    tracking_uri = cfg.get("tracking_uri")
    registry_uri = cfg.get("registry_uri")
    downgrade_unity_catalog = cfg.get("downgrade_unity_catalog")

    _validate_nonempty_str(experiment_id, "MLFlowConfig.experiment_id", is_none_allowed=True)
    _validate_nonempty_str(experiment_name, "MLFlowConfig.experiment_name", is_none_allowed=True)
    _validate_nonempty_str(run_id, "MLFlowConfig.run_id", is_none_allowed=True)
    _validate_nonempty_str(run_name, "MLFlowConfig.run_name", is_none_allowed=True)
    _validate_bool(downgrade_unity_catalog, "MLFlowConfig.downgrade_unity_catalog", is_none_allowed=True)
    _validate_uri(tracking_uri, "MLFlowConfig.tracking_uri", is_none_allowed=True)
    _validate_uri(registry_uri, "MLFlowConfig.registry_uri", is_none_allowed=True)


@dynamic_resolve
def update_mlflow_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a MLFlowConfig with the updated configuration."""
    # TODO: Is there a better way to do this?
    cfg.experiment_id = updated_cfg.get("experiment_id")
    cfg.experiment_name = updated_cfg.get("experiment_name")
    cfg.run_id = updated_cfg.get("run_id")
    cfg.run_name = updated_cfg.get("run_name")
    cfg.tracking_uri = updated_cfg.get("tracking_uri")
    cfg.registry_uri = updated_cfg.get("registry_uri")
    cfg.downgrade_unity_catalog = updated_cfg.get("downgrade_unity_catalog")
