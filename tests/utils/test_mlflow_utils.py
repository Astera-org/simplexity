"""Tests for MLflow registry URI resolution helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from mlflow.client import MlflowClient
from pytest_mock import MockerFixture

from simplexity.utils.mlflow_utils import (
    SCHEME_SEPARATOR,
    UC_PREFIX,
    WORKSPACE_PREFIX,
    get_experiment_id,
    get_run_id,
    maybe_terminate_run,
    resolve_registry_uri,
)

FILE_URI = "file:///example"
WORKSPACE_URI = WORKSPACE_PREFIX
WORKSPACE_PROFILE_URI = f"{WORKSPACE_PREFIX}{SCHEME_SEPARATOR}example_profile"
UC_URI = UC_PREFIX
UC_PROFILE_URI = f"{UC_PREFIX}{SCHEME_SEPARATOR}example_profile"


class TestResolveRegistryUri:
    """Test class for resolve_registry_uri function."""

    def test_no_uris_returns_none(self, recwarn: pytest.WarningsRecorder) -> None:
        """No URIs return None."""
        assert resolve_registry_uri() is None
        assert not recwarn.list

    @pytest.mark.parametrize("downgrade_unity_catalog", [True, False])
    def test_file_uri_is_returned_unchanged(
        self, downgrade_unity_catalog: bool, recwarn: pytest.WarningsRecorder
    ) -> None:
        """File registry URIs are returned unchanged."""
        registry_uri = FILE_URI
        assert (
            resolve_registry_uri(
                registry_uri, tracking_uri="any_tracking_uri", downgrade_unity_catalog=downgrade_unity_catalog
            )
            == registry_uri
        )
        assert not recwarn.list

    @pytest.mark.parametrize("registry_uri", [WORKSPACE_URI, WORKSPACE_PROFILE_URI])
    @pytest.mark.parametrize("downgrade_unity_catalog", [True, False])
    def test_workspace_registry_uri_is_returned_unchanged(
        self, registry_uri: str, downgrade_unity_catalog: bool, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Explicit workspace URIs are returned unchanged."""
        assert (
            resolve_registry_uri(
                registry_uri, tracking_uri="any_tracking_uri", downgrade_unity_catalog=downgrade_unity_catalog
            )
            == registry_uri
        )
        assert not recwarn.list

    @pytest.mark.parametrize("registry_uri", [UC_URI, UC_PROFILE_URI])
    def test_uc_registry_uri_is_returned_unchanged_without_downgrade(
        self, registry_uri: str, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Unity Catalog registry URIs are returned unchanged without downgrade."""
        assert (
            resolve_registry_uri(registry_uri, tracking_uri="any_tracking_uri", downgrade_unity_catalog=False)
            == registry_uri
        )
        assert not recwarn.list

    @pytest.mark.parametrize(
        ("registry_uri", "resolved_uri"),
        [
            (UC_URI, WORKSPACE_URI),
            (UC_PROFILE_URI, WORKSPACE_PROFILE_URI),
        ],
    )
    def test_uc_registry_uri_is_downgraded_to_workspace_uri(
        self, registry_uri: str, resolved_uri: str, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Unity Catalog registry URIs are downgraded to workspace URIs."""
        assert resolve_registry_uri(registry_uri, tracking_uri="any_tracking_uri") == resolved_uri
        warning = recwarn.pop(UserWarning)
        assert "Unity Catalog URI" in str(warning.message)

    def test_non_databricks_tracking_uri_ignored(self, recwarn: pytest.WarningsRecorder) -> None:
        """Non-Databricks tracking URIs are ignored."""
        assert resolve_registry_uri(tracking_uri=FILE_URI) is None
        assert not recwarn.list

    @pytest.mark.parametrize("tracking_uri", [WORKSPACE_URI, WORKSPACE_PROFILE_URI])
    @pytest.mark.parametrize("downgrade_unity_catalog", [True, False])
    def test_workspace_tracking_uri_is_returned_unchanged(
        self, tracking_uri: str, downgrade_unity_catalog: bool, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Explicit workspace URIs are returned unchanged."""
        assert (
            resolve_registry_uri(tracking_uri=tracking_uri, downgrade_unity_catalog=downgrade_unity_catalog)
            == tracking_uri
        )
        assert not recwarn.list

    @pytest.mark.parametrize("tracking_uri", [UC_URI, UC_PROFILE_URI])
    def test_uc_tracking_uri_is_returned_unchanged_without_downgrade(
        self, tracking_uri: str, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Unity Catalog registry URIs are returned unchanged without downgrade."""
        assert resolve_registry_uri(tracking_uri=tracking_uri, downgrade_unity_catalog=False) == tracking_uri
        assert not recwarn.list

    @pytest.mark.parametrize(
        ("tracking_uri", "resolved_uri"),
        [
            (UC_URI, WORKSPACE_URI),
            (UC_PROFILE_URI, WORKSPACE_PROFILE_URI),
        ],
    )
    def test_uc_tracking_uri_is_downgraded_to_workspace_uri(
        self, tracking_uri: str, resolved_uri: str, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Unity Catalog registry URIs are downgraded to workspace URIs with fallback."""
        assert resolve_registry_uri(tracking_uri=tracking_uri) == resolved_uri
        warning = recwarn.pop(UserWarning)
        assert "Unity Catalog URI" in str(warning.message)


class TestGetExperimentId:
    """Test class for get_experiment_id function."""

    def test_named_experiment_exists(self) -> None:
        """Get experiment ID."""
        existing_experiment = SimpleNamespace(experiment_id="test_experiment_id")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment_by_name.return_value = existing_experiment
        assert get_experiment_id(experiment_name="test_experiment_name", client=mock_client) == "test_experiment_id"
        mock_client.get_experiment_by_name.assert_called_once_with("test_experiment_name")
        mock_client.create_experiment.assert_not_called()

    def test_named_experiment_does_not_exist(self) -> None:
        """Get experiment ID."""
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "test_experiment_id"
        assert get_experiment_id(experiment_name="test_experiment_name", client=mock_client) == "test_experiment_id"
        mock_client.get_experiment_by_name.assert_called_once_with("test_experiment_name")
        mock_client.create_experiment.assert_called_once_with("test_experiment_name")

    def test_active_run_exists(self, mocker: MockerFixture) -> None:
        """Get experiment ID."""
        active_run = SimpleNamespace(info=SimpleNamespace(experiment_id="test_experiment_id"))
        mocker.patch("mlflow.active_run", return_value=active_run)
        assert get_experiment_id() == "test_experiment_id"

    def test_no_experiment_name_or_active_run(self) -> None:
        """Get experiment ID."""
        with pytest.raises(RuntimeError):
            get_experiment_id()


class TestGetRunId:
    """Test class for get_run_id function."""

    def test_named_run_exists(self) -> None:
        """Get run ID."""
        existing_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id"))
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.search_runs.return_value = [existing_run]
        assert (
            get_run_id(experiment_id="test_experiment_id", run_name="test_run_name", client=mock_client)
            == "test_run_id"
        )
        mock_client.search_runs.assert_called_once_with(
            experiment_ids=["test_experiment_id"], filter_string="attributes.run_name = 'test_run_name'", max_results=1
        )
        mock_client.create_run.assert_not_called()

    def test_named_run_does_not_exist(self) -> None:
        """Get run ID."""
        new_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id"))
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.search_runs.return_value = []
        mock_client.create_run.return_value = new_run
        assert (
            get_run_id(experiment_id="test_experiment_id", run_name="test_run_name", client=mock_client)
            == "test_run_id"
        )
        mock_client.search_runs.assert_called_once_with(
            experiment_ids=["test_experiment_id"], filter_string="attributes.run_name = 'test_run_name'", max_results=1
        )
        mock_client.create_run.assert_called_once_with(experiment_id="test_experiment_id", run_name="test_run_name")

    def test_active_run_exists(self, mocker: MockerFixture) -> None:
        """Get run ID."""
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", experiment_id="test_experiment_id"))
        mock_client = create_autospec(MlflowClient, instance=True)
        mocker.patch("mlflow.active_run", return_value=active_run)
        assert get_run_id(experiment_id="test_experiment_id", client=mock_client) == "test_run_id"
        mock_client.search_runs.assert_not_called()
        mock_client.create_run.assert_not_called()

    def test_active_run_experiment_id_does_not_match_experiment_id(self, mocker: MockerFixture) -> None:
        """Get run ID."""
        active_run = SimpleNamespace(
            info=SimpleNamespace(run_id="test_run_id", experiment_id="different_experiment_id")
        )
        mock_client = create_autospec(MlflowClient, instance=True)
        mocker.patch("mlflow.active_run", return_value=active_run)
        with pytest.raises(RuntimeError):
            get_run_id(experiment_id="test_experiment_id", client=mock_client)
        mock_client.search_runs.assert_not_called()
        mock_client.create_run.assert_not_called()

    def test_no_run_name_or_active_run(self, mocker: MockerFixture) -> None:
        """Get run ID."""
        new_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id"))
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.create_run.return_value = new_run
        mocker.patch("mlflow.active_run", return_value=None)
        assert get_run_id(experiment_id="test_experiment_id", client=mock_client) == "test_run_id"
        mock_client.search_runs.assert_not_called()
        mock_client.create_run.assert_called_once_with(experiment_id="test_experiment_id", run_name=None)


class TestMaybeTerminateRun:
    """Test class for maybe_terminate_run function."""

    def test_active_run(self) -> None:
        """Maybe terminate run."""
        active_run = SimpleNamespace(info=SimpleNamespace(status="RUNNING"))
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_run.return_value = active_run
        maybe_terminate_run(run_id="test_run_id", client=mock_client)
        mock_client.get_run.assert_called_once_with("test_run_id")
        mock_client.set_terminated.assert_called_once_with("test_run_id")

    @pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
    def test_active_run_is_not_running(self, status: str) -> None:
        """Maybe terminate run."""
        terminated_run = SimpleNamespace(info=SimpleNamespace(status=status))
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_run.return_value = terminated_run
        maybe_terminate_run(run_id="test_run_id", client=mock_client)
        mock_client.get_run.assert_called_once_with("test_run_id")
        mock_client.set_terminated.assert_not_called()
