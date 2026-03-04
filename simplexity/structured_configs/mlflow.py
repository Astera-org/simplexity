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

from mlflow import MlflowClient
from omegaconf import DictConfig

from simplexity.structured_configs.validation import validate_bool, validate_nonempty_str, validate_uri
from simplexity.utils.config_utils import dynamic_resolve
from simplexity.utils.mlflow_utils import get_experiment, get_run, resolve_registry_uri


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

    validate_nonempty_str(experiment_id, "MLFlowConfig.experiment_id", is_none_allowed=True)
    validate_nonempty_str(experiment_name, "MLFlowConfig.experiment_name", is_none_allowed=True)
    validate_nonempty_str(run_id, "MLFlowConfig.run_id", is_none_allowed=True)
    validate_nonempty_str(run_name, "MLFlowConfig.run_name", is_none_allowed=True)
    validate_bool(downgrade_unity_catalog, "MLFlowConfig.downgrade_unity_catalog", is_none_allowed=True)
    validate_uri(tracking_uri, "MLFlowConfig.tracking_uri", is_none_allowed=True)
    validate_uri(registry_uri, "MLFlowConfig.registry_uri", is_none_allowed=True)


@dynamic_resolve
def resolve_mlflow_config(cfg: DictConfig, *, create_if_missing: bool = True) -> None:
    """Resolve a MLFlowConfig."""
    # Resolve registry uri
    tracking_uri: str | None = cfg.get("tracking_uri")
    registry_uri: str | None = cfg.get("registry_uri")
    downgrade_unity_catalog: bool = cfg.get("downgrade_unity_catalog", True)
    cfg.registry_uri = resolve_registry_uri(
        registry_uri=registry_uri,
        tracking_uri=tracking_uri,
        downgrade_unity_catalog=downgrade_unity_catalog,
    )

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=cfg.registry_uri)

    # Resolve experiment id and name
    experiment_id: str | None = cfg.get("experiment_id")
    experiment_name: str | None = cfg.get("experiment_name")
    experiment = get_experiment(
        experiment_id=experiment_id, experiment_name=experiment_name, client=client, create_if_missing=create_if_missing
    )
    if experiment is None:
        raise ValueError(f"Experiment not found for id: {experiment_id} and name: {experiment_name}")
    cfg.experiment_id = experiment.experiment_id
    cfg.experiment_name = experiment.name

    # Resolve run id and name
    run_id: str | None = cfg.get("run_id")
    run_name: str | None = cfg.get("run_name")
    run = get_run(
        run_id=run_id,
        run_name=run_name,
        experiment_id=cfg.experiment_id,
        client=client,
        create_if_missing=create_if_missing,
    )
    if run is None:
        raise ValueError(f"Run not found for id: {run_id} and name: {run_name}")
    cfg.run_id = run.info.run_id
    cfg.run_name = run.info.run_name


@dynamic_resolve
def update_mlflow_config(cfg: DictConfig, updated_cfg: DictConfig) -> None:
    """Update a MLFlowConfig with the updated configuration."""
    cfg.merge_with(updated_cfg)
