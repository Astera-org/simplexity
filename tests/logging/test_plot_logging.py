"""Tests for plot and image logging functionality across all logger implementations."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from simplexity.logging.file_logger import FileLogger
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.logging.print_logger import PrintLogger


@pytest.fixture
def matplotlib_figure():
    """Create a reusable matplotlib figure for testing."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("Test Plot")
    yield fig
    plt.close(fig)


@pytest.fixture
def simple_matplotlib_figure():
    """Create a simple matplotlib figure for basic tests."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    yield fig
    plt.close(fig)


@pytest.fixture
def numpy_image():
    """Create a reusable numpy image array for testing."""
    return np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)


@pytest.fixture
def small_numpy_image():
    """Create a small numpy image array for testing."""
    return np.ones((50, 50, 3), dtype=np.uint8) * 100


@pytest.fixture
def tiny_numpy_image():
    """Create a tiny numpy image array for testing."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def pil_image():
    """Create a reusable PIL image for testing."""
    return Image.new("RGB", (100, 50), color="red")


@pytest.fixture
def small_pil_image():
    """Create a small PIL image for testing."""
    return Image.new("RGB", (10, 10))


@pytest.fixture
def larger_pil_image():
    """Create a larger PIL image for testing."""
    return Image.new("RGB", (20, 20))


class TestFileLoggerPlotting:
    """Tests for FileLogger figure and image logging."""

    def test_log_figure_saves_matplotlib_plot(self, matplotlib_figure, tmp_path: Path):
        """Test that log_figure saves a matplotlib figure to disk."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act
        logger.log_figure(matplotlib_figure, "test_plot.png")
        logger.close()

        # Assert
        # Verify it's a valid PNG image with expected size
        with Image.open(tmp_path / "test_plot.png") as img:
            assert img.size == (400, 300)  # 4x3 inches * 100 DPI default

        # Verify log content
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Figure saved:" in log_content
            assert "test_plot.png" in log_content

    def test_log_figure_with_kwargs(self, simple_matplotlib_figure, tmp_path: Path):
        """Test that log_figure passes kwargs to matplotlib savefig."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act
        logger.log_figure(simple_matplotlib_figure, "high_dpi.png", dpi=200, bbox_inches="tight")
        logger.close()

        # Assert
        with Image.open(tmp_path / "high_dpi.png") as img:
            # Higher DPI should result in larger image
            assert img.size[0] >= 800  # 4 inches * 200 DPI

    def test_log_image_pil_artifact_mode(self, pil_image, tmp_path: Path):
        """Test logging PIL Image in artifact mode."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act
        logger.log_image(pil_image, artifact_file="pil_test.png")
        logger.close()

        # Assert
        with Image.open(tmp_path / "pil_test.png") as img:
            assert img.size == (100, 50)

        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Image saved:" in log_content
            assert "pil_test.png" in log_content

    def test_log_image_numpy_artifact_mode(self, numpy_image, tmp_path: Path):
        """Test logging numpy array in artifact mode."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act
        logger.log_image(numpy_image, artifact_file="numpy_test.png")
        logger.close()

        # Assert
        with Image.open(tmp_path / "numpy_test.png") as img:
            assert img.size == (120, 80)  # PIL uses (width, height)

    def test_log_image_time_stepped_mode(self, small_numpy_image, tmp_path: Path):
        """Test logging image in time-stepped mode with key and step."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act
        logger.log_image(small_numpy_image, key="training_viz", step=42)
        logger.close()

        # Assert
        with Image.open(tmp_path / "training_viz_step_42.png") as img:
            assert img.size == (50, 50)

        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Time-stepped image saved:" in log_content
            assert "training_viz_step_42.png" in log_content

    def test_log_image_unsupported_type(self, tmp_path: Path):
        """Test logging unsupported image type logs error."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        unsupported_image = "not an image"

        # Act
        logger.log_image(unsupported_image, artifact_file="bad.png")  # type: ignore[arg-type]  # Intentionally testing unsupported type
        logger.close()

        # Assert
        assert not (tmp_path / "bad.png").exists()

        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "not supported for file saving" in log_content

    def test_log_image_missing_parameters_fails(self, tiny_numpy_image, tmp_path: Path):
        """Test that log_image without proper parameters logs error."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act - no artifact_file and incomplete key+step
        logger.log_image(tiny_numpy_image, key="incomplete")  # missing step
        logger.close()

        # Assert
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Image logging failed" in log_content

    def test_log_image_no_parameters_fails(self, tiny_numpy_image, tmp_path: Path):
        """Test that log_image with no parameters logs error."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))

        # Act - no parameters provided
        logger.log_image(tiny_numpy_image)  # Neither artifact_file nor key+step
        logger.close()

        # Assert
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Image logging failed - need either artifact_file or (key + step)" in log_content


class TestPrintLoggerPlotting:
    """Tests for PrintLogger figure and image logging."""

    def test_log_figure_prints_info(self, simple_matplotlib_figure, capsys):
        """Test that log_figure prints appropriate message."""
        # Arrange
        logger = PrintLogger()

        # Act
        logger.log_figure(simple_matplotlib_figure, "test.png")
        logger.close()

        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Figure NOT saved - would be: test.png (type: Figure)" in captured.out

    def test_log_image_artifact_mode_prints_info(self, tiny_numpy_image, capsys):
        """Test log_image in artifact mode prints correct message."""
        # Arrange
        logger = PrintLogger()

        # Act
        logger.log_image(tiny_numpy_image, artifact_file="test.png")
        logger.close()

        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Image NOT saved - would be artifact: test.png (type: ndarray)" in captured.out

    def test_log_image_time_stepped_mode_prints_info(self, small_pil_image, capsys):
        """Test log_image in time-stepped mode prints correct message."""
        # Arrange
        logger = PrintLogger()

        # Act
        logger.log_image(small_pil_image, key="loss_viz", step=100)
        logger.close()

        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Image NOT saved - would be key: loss_viz, step: 100 (type: Image)" in captured.out


class TestMLFlowLoggerPlotting:
    """Tests for MLFlowLogger figure and image logging."""

    @pytest.fixture(autouse=True)
    def setup_mlflow_temp_dir(self):
        """Set up temporary directory for MLflow tracking during tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set MLflow tracking URI to temp directory to avoid creating mlruns/ in project
            original_uri = os.environ.get("MLFLOW_TRACKING_URI")
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmp_dir}"
            try:
                yield
            finally:
                # Restore original URI
                if original_uri is not None:
                    os.environ["MLFLOW_TRACKING_URI"] = original_uri
                else:
                    os.environ.pop("MLFLOW_TRACKING_URI", None)

    @patch("mlflow.MlflowClient")
    def test_log_figure_calls_client_method(self, mock_client_class, simple_matplotlib_figure):
        """Test that log_figure calls the MLflow client correctly."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_client.create_run.return_value = mock_run

        logger = MLFlowLogger("test_experiment", "test_run")

        # Act
        logger.log_figure(simple_matplotlib_figure, "test.png", dpi=150)
        logger.close()

        # Assert
        mock_client.log_figure.assert_called_once_with("run_456", simple_matplotlib_figure, "test.png", dpi=150)

    @patch("mlflow.MlflowClient")
    def test_log_image_artifact_mode_calls_client(self, mock_client_class, tiny_numpy_image):
        """Test log_image in artifact mode calls MLflow client."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_client.create_run.return_value = mock_run

        logger = MLFlowLogger("test_experiment")

        # Act
        logger.log_image(tiny_numpy_image, artifact_file="image.png")
        logger.close()

        # Assert
        mock_client.log_image.assert_called_once_with(
            "run_456", tiny_numpy_image, artifact_file="image.png", key=None, step=None
        )

    @patch("mlflow.MlflowClient")
    def test_log_image_time_stepped_mode_calls_client(self, mock_client_class, larger_pil_image):
        """Test log_image in time-stepped mode calls MLflow client."""
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "exp_123"
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_client.create_run.return_value = mock_run

        logger = MLFlowLogger("test_experiment")

        # Act
        logger.log_image(larger_pil_image, key="training", step=50, timestamp=1234567890)
        logger.close()

        # Assert
        mock_client.log_image.assert_called_once_with(
            "run_456", larger_pil_image, artifact_file=None, key="training", step=50, timestamp=1234567890
        )
