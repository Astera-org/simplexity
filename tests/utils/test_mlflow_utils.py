"""Tests for MLflow registry URI resolution helpers."""

from __future__ import annotations

import pytest

from simplexity.utils.mlflow_utils import resolve_registry_uri


def test_resolve_registry_uri_prefers_explicit_workspace() -> None:
    """Explicit workspace URIs are returned unchanged."""
    assert resolve_registry_uri("databricks", "databricks") == "databricks"


def test_resolve_registry_uri_converts_uc_registry_uri(recwarn: pytest.WarningsRecorder) -> None:
    """Unity Catalog registry URIs are downgraded to workspace URIs with a warning."""
    result = resolve_registry_uri(None, "databricks-uc")
    assert result == "databricks"
    warning = recwarn.pop(UserWarning)
    assert "Unity Catalog" in str(warning.message)


def test_resolve_registry_uri_respects_disabled_fallback(recwarn: pytest.WarningsRecorder) -> None:
    """Fallback can be disabled to keep Unity Catalog URIs intact."""
    result = resolve_registry_uri(None, "databricks-uc", allow_workspace_fallback=False)
    assert result == "databricks-uc"
    assert not recwarn.list


def test_resolve_registry_uri_infers_from_tracking() -> None:
    """Databricks tracking URIs are reused for the registry by default."""
    assert resolve_registry_uri("databricks://profile", None) == "databricks://profile"


def test_resolve_registry_uri_demotes_tracking_uc(recwarn: pytest.WarningsRecorder) -> None:
    """Unity Catalog tracking URIs fall back to workspace registry URIs."""
    result = resolve_registry_uri("databricks-uc://profile", None)
    assert result == "databricks://profile"
    warning = recwarn.pop(UserWarning)
    assert "Unity Catalog tracking URI" in str(warning.message)


def test_resolve_registry_uri_tracking_fallback_toggle(recwarn: pytest.WarningsRecorder) -> None:
    """Unity Catalog tracking URIs stay untouched when fallback is disabled."""
    result = resolve_registry_uri("databricks-uc://profile", None, allow_workspace_fallback=False)
    assert result == "databricks-uc://profile"
    assert not recwarn.list


def test_resolve_registry_uri_non_databricks() -> None:
    """Non-Databricks tracking URIs leave the registry unset."""
    assert resolve_registry_uri("file:///tmp", None) is None
