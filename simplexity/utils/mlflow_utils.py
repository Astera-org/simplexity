"""Utilities for working with MLflow in different Databricks environments."""

from __future__ import annotations

import warnings
from typing import Final

_UC_PREFIX: Final = "databricks-uc"
_WORKSPACE_PREFIX: Final = "databricks"
_SCHEME_SEPARATOR: Final = "://"


def _normalize_databricks_uri(uri: str) -> str:
    """Convert Databricks Unity Catalog URIs to workspace-compatible equivalents."""
    if uri == _UC_PREFIX:
        return _WORKSPACE_PREFIX
    prefix = f"{_UC_PREFIX}{_SCHEME_SEPARATOR}"
    if uri.startswith(prefix):
        suffix = uri.split(_SCHEME_SEPARATOR, 1)[1]
        return f"{_WORKSPACE_PREFIX}{_SCHEME_SEPARATOR}{suffix}"
    return uri


def resolve_registry_uri(
    registry_uri: str | None,
    tracking_uri: str | None,
    *,
    allow_workspace_fallback: bool = True,
) -> str | None:
    """Determine a workspace model registry URI for MLflow operations.

    - If an explicit registry URI is provided, convert Unity Catalog URIs to their
      workspace equivalents while warning the caller about the downgrade.
    - If no registry URI is provided, infer one from a Databricks tracking URI.
    - For non-Databricks configurations, return ``None`` so MLflow uses its defaults.
    """
    if registry_uri:
        if allow_workspace_fallback:
            normalized = _normalize_databricks_uri(registry_uri)
            if normalized != registry_uri:
                warnings.warn(
                    (
                        f"Unity Catalog registry URI '{registry_uri}' is not supported by this environment; "
                        f"using workspace registry URI '{normalized}' instead."
                    ),
                    stacklevel=2,
                )
            return normalized
        return registry_uri

    if tracking_uri:
        normalized = _normalize_databricks_uri(tracking_uri)
        if normalized.startswith((_WORKSPACE_PREFIX, _UC_PREFIX)):
            if normalized != tracking_uri:
                warnings.warn(
                    (
                        f"Unity Catalog tracking URI '{tracking_uri}' detected; "
                        f"falling back to workspace registry URI '{normalized}'."
                    ),
                    stacklevel=2,
                )
            return normalized

    return None


__all__ = ["resolve_registry_uri"]
