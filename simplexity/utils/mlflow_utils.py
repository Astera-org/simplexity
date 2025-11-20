"""Utilities for working with MLflow in different Databricks environments."""

from __future__ import annotations

import configparser
import logging
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mlflow

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Experiment, Run

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


@contextmanager
def set_mlflow_uris(tracking_uri: str | None, registry_uri: str | None) -> Generator[None, None, None]:
    """Set the tracking and registry URIs for the current context."""
    original_tracking_uri = mlflow.get_tracking_uri()
    original_registry_uri = mlflow.get_registry_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    yield
    mlflow.set_tracking_uri(original_tracking_uri)
    mlflow.set_registry_uri(original_registry_uri)


def _get_client(client: MlflowClient | None = None) -> MlflowClient:
    """Get an MLflow client."""
    return mlflow.MlflowClient() if client is None else client


def get_experiment_by_id(experiment_id: str, client: MlflowClient | None = None) -> Experiment:
    """Get an experiment by id."""
    client = _get_client(client)
    experiment = client.get_experiment(experiment_id)
    if not experiment:
        raise RuntimeError(f"Experiment with id '{experiment_id}' does not exist")
    SIMPLEXITY_LOGGER.info("[mlflow] experiment with id '%s' exists with name: '%s'", experiment_id, experiment.name)
    return experiment


def get_experiment_by_name(
    experiment_name: str, client: MlflowClient | None = None, *, create_if_missing: bool = True
) -> Experiment | None:
    """Get an experiment by name."""
    client = _get_client(client)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        SIMPLEXITY_LOGGER.info(
            "[mlflow] experiment with name '%s' already exists with id: %s",
            experiment_name,
            experiment.experiment_id,
        )
        return experiment
    SIMPLEXITY_LOGGER.info("[mlflow] experiment with name '%s' does not exist", experiment_name)
    if create_if_missing:
        experiment_id = client.create_experiment(experiment_name)
        SIMPLEXITY_LOGGER.info("[mlflow] experiment with name '%s' created with id: %s", experiment_name, experiment_id)
        return client.get_experiment(experiment_id)
    return None


def get_active_experiment(client: MlflowClient | None = None) -> Experiment | None:
    """Get the active experiment."""
    active_run = mlflow.active_run()
    if active_run:
        SIMPLEXITY_LOGGER.info("[mlflow] active run exists with experiment id: %s", active_run.info.experiment_id)
        client = _get_client(client)
        return client.get_experiment(active_run.info.experiment_id)
    SIMPLEXITY_LOGGER.info("[mlflow] no active run found")
    return None


def get_experiment(
    experiment_id: str | None = None,
    *,
    experiment_name: str | None = None,
    client: MlflowClient | None = None,
    create_if_missing: bool = True,
) -> Experiment | None:
    """Get an MLflow experiment."""
    if experiment_id:
        experiment = get_experiment_by_id(experiment_id, client)
        if experiment_name is not None and experiment.name != experiment_name:
            raise RuntimeError(
                f"Experiment with id '{experiment_id}' has name '{experiment.name}' but expected '{experiment_name}'"
            )
        return experiment
    if experiment_name:
        return get_experiment_by_name(experiment_name, client, create_if_missing=create_if_missing)
    return get_active_experiment(client)


def get_run_by_id(run_id: str, client: MlflowClient | None = None) -> Run:
    """Get a run by id."""
    client = _get_client(client)
    run = client.get_run(run_id)
    if not run:
        raise RuntimeError(f"Run with id '{run_id}' does not exist")
    SIMPLEXITY_LOGGER.info("[mlflow] run with id '%s' exists with name: '%s'", run_id, run.info.run_name)
    return run


def get_run_by_name(
    run_name: str, experiment_id: str, client: MlflowClient | None = None, *, create_if_missing: bool = True
) -> Run | None:
    """Get a run by name."""
    client = _get_client(client)
    runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string=f"attributes.run_name = '{run_name}'", max_results=1
    )
    if runs:
        SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' exists with id: %s", run_name, runs[0].info.run_id)
        return runs[0]
    SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' does not exist", run_name)
    if create_if_missing:
        run = client.create_run(experiment_id=experiment_id, run_name=run_name)
        SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' created with id: %s", run_name, run.info.run_id)
        return run
    return None


def get_active_run() -> Run | None:
    """Get the active run."""
    active_run = mlflow.active_run()
    if active_run:
        SIMPLEXITY_LOGGER.info("[mlflow] active run exists with id: %s", active_run.info.run_id)
        return active_run
    SIMPLEXITY_LOGGER.info("[mlflow] no active run found")
    return None


def get_run(
    run_id: str | None = None,
    run_name: str | None = None,
    experiment_id: str | None = None,
    client: MlflowClient | None = None,
    *,
    create_if_missing: bool = True,
) -> Run | None:
    """Get an MLflow run."""
    if run_id:
        run = get_run_by_id(run_id, client)
        if run_name is not None and run.info.run_name != run_name:
            raise RuntimeError(f"Run with id '{run_id}' has name '{run.info.run_name}' but expected '{run_name}'")
        return run
    if run_name:
        if experiment_id is None:
            raise RuntimeError("Experiment id is required when getting a run by name")
        return get_run_by_name(run_name, experiment_id, client, create_if_missing=create_if_missing)
    active_run = get_active_run()
    if active_run:
        if experiment_id is not None and active_run.info.experiment_id != experiment_id:
            raise RuntimeError(
                f"Active run experiment id {active_run.info.experiment_id} does not match experiment id {experiment_id}"
            )
        return active_run
    if create_if_missing:
        if experiment_id is None:
            raise RuntimeError("Experiment id is required when creating a run")
        client = _get_client(client)
        run = client.create_run(experiment_id=experiment_id)
        SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' created with id: %s", run_name, run.info.run_id)
        return run
    return None


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
