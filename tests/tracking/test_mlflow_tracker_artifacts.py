"""Tests for artifact logging functionality for MlflowTracker."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from simplexity.tracking.mlflow_tracker import MlflowTracker


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {"key": "value", "number": 42, "list": [1, 2, 3]}


class TestMlflowTrackerArtifacts:
    """Tests for MlflowTracker artifact logging."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_temp_dir(self):
        """Set up temporary directory for MLflow tracking during tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_uri = os.environ.get("MLFLOW_TRACKING_URI")
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_dir}"
            try:
                yield
            finally:
                if original_uri is not None:
                    os.environ["MLFLOW_TRACKING_URI"] = original_uri
                else:
                    os.environ.pop("MLFLOW_TRACKING_URI", None)

    @pytest.fixture
    def mock_mlflow_tracker(self):
        """Create a mocked MlflowTracker for testing."""
        with patch("mlflow.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            # Mock experiment and run retrieval/creation
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = "exp_123"
            mock_experiment.name = "test_experiment"
            mock_client.get_experiment_by_name.return_value = None
            mock_client.create_experiment.return_value = "exp_123"
            mock_client.get_experiment.return_value = mock_experiment  # For get_experiment helper

            mock_run = MagicMock()
            mock_run.info.run_id = "run_456"
            mock_run.info.run_name = "test_run"
            mock_client.create_run.return_value = mock_run
            mock_client.get_run.return_value = mock_run

            tracker = MlflowTracker(experiment_name="test_experiment")
            yield tracker, mock_client

    def test_log_artifact_calls_client(self, mock_mlflow_tracker):
        """Test that log_artifact calls the MLflow client correctly."""
        tracker, mock_client = mock_mlflow_tracker

        tracker.log_artifact("/path/to/file.txt", "artifacts/file.txt")
        tracker.close()

        mock_client.log_artifact.assert_called_once_with("run_456", "/path/to/file.txt", "artifacts/file.txt")

    def test_log_artifact_without_artifact_path(self, mock_mlflow_tracker):
        """Test log_artifact without custom artifact path."""
        tracker, mock_client = mock_mlflow_tracker

        tracker.log_artifact("/path/to/model.pkl")
        tracker.close()

        mock_client.log_artifact.assert_called_once_with("run_456", "/path/to/model.pkl", None)

    def test_log_json_artifact_calls_client(self, mock_mlflow_tracker, sample_json_data):
        """Test that log_json_artifact creates temp file and calls client."""
        tracker, mock_client = mock_mlflow_tracker

        tracker.log_json_artifact(sample_json_data, "metrics.json")
        tracker.close()

        mock_client.log_artifact.assert_called_once()
        call_args = mock_client.log_artifact.call_args
        assert call_args[0][0] == "run_456"  # run_id
        assert call_args[0][1].endswith("metrics.json")  # temp file path
        assert len(call_args[0]) == 2
