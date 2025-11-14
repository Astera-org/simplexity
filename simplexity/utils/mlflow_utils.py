"""Utilities for working with MLflow in different Databricks environments."""

from __future__ import annotations

import configparser
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mlflow

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Run

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")

UC_PREFIX: Final = "databricks-uc"
WORKSPACE_PREFIX: Final = "databricks"
SCHEME_SEPARATOR: Final = "://"
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.ini"


TERMINAL_STATES = ["FINISHED", "FAILED", "KILLED"]


def get_databricks_host() -> str | None:
    """Load configuration from config.ini file."""
    if not _CONFIG_PATH.exists():
        warnings.warn(
            f"[mlflow] configuration file not found at {_CONFIG_PATH}",
            stacklevel=2,
        )
        return None

    config = configparser.ConfigParser()
    try:
        config.read(_CONFIG_PATH)
        if "databricks" not in config:
            raise configparser.NoSectionError("databricks")
        if "host" not in config["databricks"]:
            raise configparser.NoOptionError("host", "databricks")

        host = config["databricks"]["host"]
        SIMPLEXITY_LOGGER.info("[mlflow] databricks host: %s", host)
        return host
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        warnings.warn(
            f"[mlflow] error reading configuration: {e}",
            stacklevel=2,
        )
        return None


def resolve_registry_uri(
    registry_uri: str | None = None,
    *,
    tracking_uri: str | None = None,
    downgrade_unity_catalog: bool = True,
) -> str | None:
    """Determine a workspace model registry URI for MLflow operations."""

    def convert_uri(uri: str) -> str:
        """Convert Databricks Unity Catalog URIs to workspace-compatible equivalents."""
        prefix, sep, suffix = uri.partition(SCHEME_SEPARATOR)
        if prefix == UC_PREFIX:
            normalized_uri = f"{WORKSPACE_PREFIX}{sep}{suffix}"
            warnings.warn(
                (
                    f"[mlflow] Unity Catalog URI '{uri}' is not supported by this environment; "
                    f"using workspace URI '{normalized_uri}' instead."
                ),
                stacklevel=3,
            )
            return normalized_uri
        return uri

    if registry_uri:
        if downgrade_unity_catalog:
            registry_uri = convert_uri(registry_uri)
        SIMPLEXITY_LOGGER.info("[mlflow] registry uri: %s", registry_uri)
        return registry_uri

    if tracking_uri and tracking_uri.startswith("databricks"):
        if downgrade_unity_catalog:
            tracking_uri = convert_uri(tracking_uri)
        SIMPLEXITY_LOGGER.info("[mlflow] registry uri defaulting to tracking uri: %s", tracking_uri)
        return tracking_uri

    SIMPLEXITY_LOGGER.info("[mlflow] no registry uri or tracking uri found")
    return None


def get_experiment_id(
    experiment_name: str | None = None,
    client: MlflowClient | None = None,
) -> str:
    """Get the experiment id of an MLflow experiment."""
    if experiment_name:
        client = mlflow.MlflowClient() if client is None else client
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            SIMPLEXITY_LOGGER.info(
                "[mlflow] experiment with name '%s' already exists with id: %s",
                experiment_name,
                experiment.experiment_id,
            )
            return experiment.experiment_id
        experiment_id = client.create_experiment(experiment_name)
        SIMPLEXITY_LOGGER.info("[mlflow] experiment with name '%s' created with id: %s", experiment_name, experiment_id)
        return experiment_id
    active_run = mlflow.active_run()
    if active_run:
        SIMPLEXITY_LOGGER.info("[mlflow] active run exists with experiment id: %s", active_run.info.experiment_id)
        return active_run.info.experiment_id
    raise RuntimeError("No experiment name or active run found")


def get_run_id(
    experiment_id: str,
    run_name: str | None = None,
    client: MlflowClient | None = None,
) -> str:
    """Get the run id of an MLflow run."""
    client = mlflow.MlflowClient() if client is None else client
    if run_name:
        runs = client.search_runs(
            experiment_ids=[experiment_id], filter_string=f"attributes.run_name = '{run_name}'", max_results=1
        )
        if runs:
            SIMPLEXITY_LOGGER.info(
                "[mlflow] run with name '%s' already exists with id: %s", run_name, runs[0].info.run_id
            )
            return runs[0].info.run_id
        run: Run = client.create_run(experiment_id=experiment_id, run_name=run_name)
        SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' created with id: %s", run_name, run.info.run_id)
        return run.info.run_id
    active_run = mlflow.active_run()
    if active_run:
        SIMPLEXITY_LOGGER.info("[mlflow] active run exists with id: %s", active_run.info.run_id)
        if active_run.info.experiment_id != experiment_id:
            raise RuntimeError(
                f"Active run experiment id {active_run.info.experiment_id} does not match experiment id {experiment_id}"
            )
        return active_run.info.run_id
    run = client.create_run(experiment_id=experiment_id, run_name=run_name)
    SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' created with id: %s", run_name, run.info.run_id)
    return run.info.run_id


def maybe_terminate_run(run_id: str, client: MlflowClient | None = None) -> None:
    """Terminate an MLflow run."""
    client = mlflow.MlflowClient() if client is None else client
    status = client.get_run(run_id).info.status
    if status not in TERMINAL_STATES:
        SIMPLEXITY_LOGGER.info("[mlflow] terminating run with id: %s", run_id)
        client.set_terminated(run_id)
    else:
        SIMPLEXITY_LOGGER.debug("[mlflow] run with id: %s is already terminated with status: %s", run_id, status)


__all__ = ["resolve_registry_uri"]
