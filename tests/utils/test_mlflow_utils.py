"""Tests for MLflow registry URI resolution helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import call, create_autospec

import mlflow
import pytest
from mlflow.client import MlflowClient
from pytest_mock import MockerFixture

from simplexity.utils.mlflow_utils import (
    SCHEME_SEPARATOR,
    UC_PREFIX,
    WORKSPACE_PREFIX,
    get_active_experiment,
    get_active_run,
    get_experiment,
    get_experiment_by_id,
    get_experiment_by_name,
    get_run,
    get_run_by_id,
    get_run_by_name,
    maybe_terminate_run,
    resolve_registry_uri,
    set_mlflow_uris,
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


class TestSetMlflowUris:  # pylint: disable=too-few-public-methods
    """Test class for set_mlflow_uris function."""

    def test_set_mlflow_uris(self) -> None:
        """Set MLflow URIs."""
        mlflow.set_tracking_uri("original_tracking_uri")
        mlflow.set_registry_uri("original_registry_uri")
        assert mlflow.get_tracking_uri() == "original_tracking_uri"
        assert mlflow.get_registry_uri() == "original_registry_uri"
        with set_mlflow_uris(tracking_uri="test_tracking_uri", registry_uri="test_registry_uri"):
            assert mlflow.get_tracking_uri() == "test_tracking_uri"
            assert mlflow.get_registry_uri() == "test_registry_uri"
        assert mlflow.get_tracking_uri() == "original_tracking_uri"
        assert mlflow.get_registry_uri() == "original_registry_uri"


class TestGetExperiment:
    """Test class for get_experiment_id function."""

    def test_get_by_id_exists(self, mocker: MockerFixture) -> None:
        """Get experiment by ID.

        If an experiment exists with the given ID,
        it is returned.
        Applies to get_experiment_by_id and get_experiment.
        """
        existing_experiment = SimpleNamespace(experiment_id="test_experiment_id", name="test_experiment_name")

        mock_ingo = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment.return_value = existing_experiment

        assert get_experiment_by_id("test_experiment_id", client=mock_client) == existing_experiment
        mock_ingo.assert_called_once_with(
            "[mlflow] experiment with id '%s' exists with name: '%s'",
            "test_experiment_id",
            "test_experiment_name",
        )

        mock_client.reset_mock()
        mock_ingo.reset_mock()

        assert get_experiment(experiment_id="test_experiment_id", client=mock_client) == existing_experiment
        mock_ingo.assert_called_once_with(
            "[mlflow] experiment with id '%s' exists with name: '%s'",
            "test_experiment_id",
            "test_experiment_name",
        )

    def test_get_by_id_does_not_exist(self) -> None:
        """Get experiment by ID.

        If an experiment does not exist with the given ID, a RuntimeError is raised.
        Applies to get_experiment_by_id and get_experiment.
        """
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment.return_value = None

        with pytest.raises(RuntimeError):
            get_experiment_by_id("test_experiment_id", client=mock_client)

        mock_client.reset_mock()

        with pytest.raises(RuntimeError):
            get_experiment(experiment_id="test_experiment_id", client=mock_client)

    def test_conflicting_id_and_name(self) -> None:
        """Get experiment by ID.

        If an experiment exists with the given ID but has a different name than the given name,
        a RuntimeError is raised.
        Applies to get_experiment.
        """
        existing_experiment = SimpleNamespace(experiment_id="test_experiment_id", name="test_experiment_name")

        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment.return_value = existing_experiment

        with pytest.raises(RuntimeError):
            get_experiment(experiment_id="test_experiment_id", experiment_name="different_name", client=mock_client)

    def test_get_by_name_exists(self, mocker: MockerFixture) -> None:
        """Get experiment ID.

        If an experiment exists with the given name, it is returned.
        Applies to get_experiment_by_name and get_experiment (without experiment_id argument).
        """
        existing_experiment = SimpleNamespace(experiment_id="test_experiment_id", name="test_experiment_name")

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment_by_name.return_value = existing_experiment

        assert get_experiment_by_name("test_experiment_name", client=mock_client) == existing_experiment
        mock_info.assert_called_once_with(
            "[mlflow] experiment with name '%s' already exists with id: %s",
            "test_experiment_name",
            "test_experiment_id",
        )

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_experiment(experiment_name="test_experiment_name", client=mock_client) == existing_experiment
        mock_info.assert_called_once_with(
            "[mlflow] experiment with name '%s' already exists with id: %s",
            "test_experiment_name",
            "test_experiment_id",
        )

    def test_get_by_name_creates_missing(self, mocker: MockerFixture) -> None:
        """Get experiment ID.

        If an existing experiment does not exist with the given name,
        by default a new experiment is created with that name and returned.
        Applies to get_experiment_by_name and get_experiment (without experiment_id argument).
        """
        created_experiment = SimpleNamespace(experiment_id="test_experiment_id", name="test_experiment_name")

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "test_experiment_id"
        mock_client.get_experiment.return_value = created_experiment

        assert get_experiment_by_name("test_experiment_name", client=mock_client) == created_experiment
        mock_client.create_experiment.assert_called_once_with("test_experiment_name")
        mock_info.assert_has_calls(
            [
                call("[mlflow] experiment with name '%s' does not exist", "test_experiment_name"),
                call(
                    "[mlflow] experiment with name '%s' created with id: %s",
                    "test_experiment_name",
                    "test_experiment_id",
                ),
            ]
        )

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_experiment(experiment_name="test_experiment_name", client=mock_client) == created_experiment
        mock_client.create_experiment.assert_called_once_with("test_experiment_name")
        mock_info.assert_has_calls(
            [
                call("[mlflow] experiment with name '%s' does not exist", "test_experiment_name"),
                call(
                    "[mlflow] experiment with name '%s' created with id: %s",
                    "test_experiment_name",
                    "test_experiment_id",
                ),
            ]
        )

    def test_get_name_does_not_create_missing(self, mocker: MockerFixture) -> None:
        """Get experiment ID.

        If an existing experiment does not exist with the given name and create_if_missing is False,
        nothing is returned.
        Applies to get_experiment_by_name and get_experiment (without experiment_id argument).
        """
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment_by_name.return_value = None

        assert get_experiment_by_name("test_experiment_name", client=mock_client, create_if_missing=False) is None
        mock_info.assert_called_once_with("[mlflow] experiment with name '%s' does not exist", "test_experiment_name")

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert (
            get_experiment(experiment_name="test_experiment_name", client=mock_client, create_if_missing=False) is None
        )
        mock_info.assert_called_once_with("[mlflow] experiment with name '%s' does not exist", "test_experiment_name")

    def test_get_active_experiment(self, mocker: MockerFixture) -> None:
        """Get active experiment.

        If an active run exists, the experiment of that run is returned.
        Applies to get_active_experiment and get_experiment (without experiment_id or experiment_name arguments).
        """
        active_run = SimpleNamespace(info=SimpleNamespace(experiment_id="test_experiment_id"))
        active_experiment = SimpleNamespace(experiment_id="test_experiment_id")

        mocker.patch("mlflow.active_run", return_value=active_run)
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_experiment.return_value = active_experiment

        assert get_active_experiment(client=mock_client) == active_experiment
        mock_info.assert_called_once_with("[mlflow] active run exists with experiment id: %s", "test_experiment_id")

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_experiment(client=mock_client) == active_experiment
        mock_info.assert_called_once_with("[mlflow] active run exists with experiment id: %s", "test_experiment_id")

    def test_no_active_run(self, mocker: MockerFixture) -> None:
        """Get active experiment.

        If no active run exists, nothing is returned.
        Applies to get_active_experiment and get_experiment (without experiment_id or experiment_name arguments).
        """
        mocker.patch("mlflow.active_run", return_value=None)
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)

        assert get_active_experiment(client=mock_client) is None
        mock_info.assert_called_once_with("[mlflow] no active run found")

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_experiment(client=mock_client) is None
        mock_info.assert_called_once_with("[mlflow] no active run found")


class TestGetRun:
    """Test class for get_run_id function."""

    def test_get_by_id_exists(self, mocker: MockerFixture) -> None:
        """Get run by ID.

        If a run exists with the given ID, it is returned.
        Applies to get_run_by_id and get_run.
        """
        existing_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", run_name="test_run_name"))

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_run.return_value = existing_run

        assert get_run_by_id("test_run_id", client=mock_client) == existing_run
        mock_info.assert_called_once_with(
            "[mlflow] run with id '%s' exists with name: '%s'",
            "test_run_id",
            "test_run_name",
        )

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_run(run_id="test_run_id", client=mock_client) == existing_run
        mock_info.assert_called_once_with(
            "[mlflow] run with id '%s' exists with name: '%s'",
            "test_run_id",
            "test_run_name",
        )

    def test_get_by_id_does_not_exist(self) -> None:
        """Get run by ID.

        If a run does not exist with the given ID, a RuntimeError is raised.
        Applies to get_run_by_id and get_run.
        """
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_run.return_value = None

        with pytest.raises(RuntimeError):
            get_run_by_id("test_run_id", client=mock_client)

        mock_client.reset_mock()

        with pytest.raises(RuntimeError):
            get_run(run_id="test_run_id", client=mock_client)

    def test_conflicting_id_and_name(self) -> None:
        """Get run by ID.

        If a run exists with the given ID but has a different name than the given name,
        a RuntimeError is raised.
        Applies to get_run.
        """
        existing_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", run_name="test_run_name"))

        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.get_run.return_value = existing_run

        with pytest.raises(RuntimeError):
            get_run(run_id="test_run_id", run_name="different_name", client=mock_client)

    def test_get_by_name_exists(self, mocker: MockerFixture) -> None:
        """Get run by name.

        If a run exists with the given name, it is returned.
        Applies to get_run_by_name and get_run (without run_id argument).
        """
        existing_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", run_name="test_run_name"))

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.search_runs.return_value = [existing_run]

        assert get_run_by_name("test_run_name", "test_experiment_id", client=mock_client) == existing_run
        mock_info.assert_called_once_with(
            "[mlflow] run with name '%s' exists with id: %s",
            "test_run_name",
            "test_run_id",
        )

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_run(run_name="test_run_name", experiment_id="test_experiment_id", client=mock_client) == existing_run
        mock_info.assert_called_once_with(
            "[mlflow] run with name '%s' exists with id: %s",
            "test_run_name",
            "test_run_id",
        )

    def test_get_by_name_creates_missing(self, mocker: MockerFixture) -> None:
        """Get run by name.

        If a run does not exist with the given name,
        by default a new run is created with that name and returned.
        Applies to get_run_by_name and get_run (without run_id argument).
        """
        created_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", run_name="test_run_name"))

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.search_runs.return_value = []
        mock_client.create_run.return_value = created_run

        assert get_run_by_name("test_run_name", "test_experiment_id", client=mock_client) == created_run
        mock_client.create_run.assert_called_once_with(experiment_id="test_experiment_id", run_name="test_run_name")
        mock_info.assert_has_calls(
            [
                call("[mlflow] run with name '%s' does not exist", "test_run_name"),
                call("[mlflow] run with name '%s' created with id: %s", "test_run_name", "test_run_id"),
            ]
        )

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert get_run(run_name="test_run_name", experiment_id="test_experiment_id", client=mock_client) == created_run
        mock_client.create_run.assert_called_once_with(experiment_id="test_experiment_id", run_name="test_run_name")
        mock_info.assert_has_calls(
            [
                call("[mlflow] run with name '%s' does not exist", "test_run_name"),
                call("[mlflow] run with name '%s' created with id: %s", "test_run_name", "test_run_id"),
            ]
        )

    def test_get_name_does_not_create_missing(self, mocker: MockerFixture) -> None:
        """Get run by name.

        If a run does not exist with the given name and create_if_missing is False,
        nothing is returned.
        Applies to get_run_by_name and get_run (without run_id argument).
        """
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.search_runs.return_value = []

        assert (
            get_run_by_name("test_run_name", "test_experiment_id", client=mock_client, create_if_missing=False) is None
        )
        mock_client.create_run.assert_not_called()
        mock_info.assert_called_once_with("[mlflow] run with name '%s' does not exist", "test_run_name")

        mock_client.reset_mock()
        mock_info.reset_mock()

        assert (
            get_run(
                run_name="test_run_name",
                experiment_id="test_experiment_id",
                client=mock_client,
                create_if_missing=False,
            )
            is None
        )
        mock_client.create_run.assert_not_called()
        mock_info.assert_called_once_with("[mlflow] run with name '%s' does not exist", "test_run_name")

    def test_get_name_requires_experiment_id(self) -> None:
        """Get run by name.

        If a run is requested by name without an experiment_id, a RuntimeError is raised.
        Applies to get_run.
        """
        mock_client = create_autospec(MlflowClient, instance=True)

        with pytest.raises(RuntimeError):
            get_run(run_name="test_run_name", client=mock_client)

    def test_get_active_run(self, mocker: MockerFixture) -> None:
        """Get active run.

        If an active run exists, it is returned.
        Applies to get_active_run and get_run (without run_id or run_name arguments).
        """
        active_run = SimpleNamespace(
            info=SimpleNamespace(run_id="test_run_id", run_name="test_run_name", experiment_id="test_experiment_id")
        )

        mocker.patch("mlflow.active_run", return_value=active_run)
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")

        assert get_active_run() == active_run
        mock_info.assert_called_once_with("[mlflow] active run exists with id: %s", "test_run_id")

        mock_info.reset_mock()

        assert get_run() == active_run
        mock_info.assert_called_once_with("[mlflow] active run exists with id: %s", "test_run_id")

    def test_active_run_experiment_mismatch(self, mocker: MockerFixture) -> None:
        """Get active run.

        If an active run exists but the experiment_id does not match the requested experiment,
        a RuntimeError is raised.
        Applies to get_run.
        """
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", experiment_id="actual_experiment_id"))
        mocker.patch("mlflow.active_run", return_value=active_run)

        with pytest.raises(RuntimeError):
            get_run(experiment_id="different_experiment_id")

    def test_create_run_when_missing(self, mocker: MockerFixture) -> None:
        """Create run if missing.

        If no active run exists and create_if_missing is True,
        a new run is created with the provided experiment_id.
        Applies to get_run.
        """
        mocker.patch("simplexity.utils.mlflow_utils.get_active_run", return_value=None)
        created_run = SimpleNamespace(info=SimpleNamespace(run_id="test_run_id", run_name=None))

        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")
        mock_client = create_autospec(MlflowClient, instance=True)
        mock_client.create_run.return_value = created_run

        assert get_run(experiment_id="test_experiment_id", client=mock_client) == created_run
        mock_client.create_run.assert_called_once_with(experiment_id="test_experiment_id")
        mock_info.assert_called_once_with("[mlflow] run with name '%s' created with id: %s", None, "test_run_id")

    def test_create_run_requires_experiment_id(self, mocker: MockerFixture) -> None:
        """Create run if missing.

        If no active run exists and create_if_missing is True but experiment_id is missing,
        a RuntimeError is raised.
        Applies to get_run.
        """
        mocker.patch("simplexity.utils.mlflow_utils.get_active_run", return_value=None)

        with pytest.raises(RuntimeError):
            get_run()

    def test_no_active_run(self, mocker: MockerFixture) -> None:
        """Get active run.

        If no active run exists and create_if_missing is False,
        nothing is returned.
        Applies to get_active_run and get_run.
        """
        mocker.patch("mlflow.active_run", return_value=None)
        mock_info = mocker.patch("simplexity.utils.mlflow_utils.SIMPLEXITY_LOGGER.info")

        assert get_active_run() is None
        mock_info.assert_called_once_with("[mlflow] no active run found")

        mock_info.reset_mock()

        assert get_run(create_if_missing=False) is None
        mock_info.assert_called_once_with("[mlflow] no active run found")


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
