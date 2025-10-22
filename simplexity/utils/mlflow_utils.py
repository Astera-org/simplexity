"""Utilities for working with MLflow in different Databricks environments."""

from __future__ import annotations

import configparser
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mlflow import MlflowClient

UC_PREFIX: Final = "databricks-uc"
WORKSPACE_PREFIX: Final = "databricks"
SCHEME_SEPARATOR: Final = "://"
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.ini"


def get_databricks_host() -> str | None:
    """Load configuration from config.ini file."""
    if not _CONFIG_PATH.exists():
        print(f"Error: Configuration file not found at {_CONFIG_PATH}")
        print("Please create a config.ini file based on config.ini.example")
        return None

    config = configparser.ConfigParser()
    try:
        config.read(_CONFIG_PATH)
        if "databricks" not in config:
            raise configparser.NoSectionError("databricks")
        if "host" not in config["databricks"]:
            raise configparser.NoOptionError("host", "databricks")

        return config["databricks"]["host"]
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading configuration: {e}")
        print("Please ensure config.ini has a [databricks] section with a 'host' key")
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
                    f"Unity Catalog URI '{uri}' is not supported by this environment; "
                    f"using workspace URI '{normalized_uri}' instead."
                ),
                stacklevel=3,
            )
            return normalized_uri
        return uri

    if registry_uri:
        if downgrade_unity_catalog:
            return convert_uri(registry_uri)
        return registry_uri

    if tracking_uri and tracking_uri.startswith("databricks"):
        if downgrade_unity_catalog:
            return convert_uri(tracking_uri)
        return tracking_uri

    return None


def maybe_terminate_run(client: MlflowClient, run_id: str) -> None:
    """Terminate an MLflow run."""
    terminal_statuses = ["FINISHED", "FAILED", "KILLED"]
    status = client.get_run(run_id).info.status
    if status not in terminal_statuses:
        client.set_terminated(run_id)


__all__ = ["resolve_registry_uri"]
