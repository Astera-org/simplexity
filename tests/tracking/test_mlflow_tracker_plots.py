"""Tests for plot and image logging functionality for MlflowTracker."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from simplexity.tracking.mlflow_tracker import MlflowTracker


@pytest.fixture
def simple_matplotlib_figure():
    """Create a simple matplotlib figure for basic tests."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    yield fig
    plt.close(fig)


@pytest.fixture
def tiny_numpy_image():
    """Create a tiny numpy image array for testing."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def larger_pil_image():
    """Create a larger PIL image for testing."""
    return Image.new("RGB", (20, 20))


class TestMlflowTrackerPlotting:
    """Tests for MlflowTracker figure and image logging."""

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

    @patch("mlflow.MlflowClient")
    def test_log_figure_calls_client_method(self, mock_client_class, simple_matplotlib_figure):
        """Test that log_figure calls the MLflow client correctly."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock experiment retrieval/creation
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.name = "test_experiment"
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_client.get_experiment.return_value = mock_experiment

        mock_client.search_runs.return_value = []
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_run.info.run_name = "test_run"
        mock_client.create_run.return_value = mock_run
        mock_client.get_run.return_value = mock_run

        tracker = MlflowTracker(experiment_name="test_experiment", run_name="test_run")

        tracker.log_figure(simple_matplotlib_figure, "test.png", dpi=150)
        tracker.close()

        mock_client.log_figure.assert_called_once_with("run_456", simple_matplotlib_figure, "test.png", dpi=150)

    @patch("mlflow.MlflowClient")
    def test_log_image_artifact_mode_calls_client(self, mock_client_class, tiny_numpy_image):
        """Test log_image in artifact mode calls MLflow client."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.name = "test_experiment"
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_client.get_experiment.return_value = mock_experiment

        mock_client.search_runs.return_value = []
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_run.info.run_name = "test_run"
        mock_client.create_run.return_value = mock_run
        mock_client.get_run.return_value = mock_run

        tracker = MlflowTracker(experiment_name="test_experiment")

        tracker.log_image(tiny_numpy_image, artifact_file="image.png")
        tracker.close()

        mock_client.log_image.assert_called_once_with(
            "run_456", tiny_numpy_image, artifact_file="image.png", key=None, step=None
        )

    @patch("mlflow.MlflowClient")
    def test_log_image_time_stepped_mode_calls_client(self, mock_client_class, larger_pil_image):
        """Test log_image in time-stepped mode calls MLflow client."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_experiment.name = "test_experiment"
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_client.get_experiment.return_value = mock_experiment

        mock_client.search_runs.return_value = []
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_run.info.run_name = "test_run"
        mock_client.create_run.return_value = mock_run
        mock_client.get_run.return_value = mock_run

        tracker = MlflowTracker(experiment_name="test_experiment")

        tracker.log_image(larger_pil_image, key="training", step=50, timestamp=1234567890)
        tracker.close()

        mock_client.log_image.assert_called_once_with(
            "run_456", larger_pil_image, artifact_file=None, key="training", step=50, timestamp=1234567890
        )
