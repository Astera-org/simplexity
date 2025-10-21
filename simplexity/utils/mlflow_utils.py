"""Utilities for working with MLflow in different Databricks environments."""

from __future__ import annotations

import warnings
from typing import Final

UC_PREFIX: Final = "databricks-uc"
WORKSPACE_PREFIX: Final = "databricks"
SCHEME_SEPARATOR: Final = "://"


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


__all__ = ["resolve_registry_uri"]
