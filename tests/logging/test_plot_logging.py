"""Tests for plot and image logging functionality across all logger implementations."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from simplexity.logging.file_logger import FileLogger
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.logging.print_logger import PrintLogger


class TestFileLoggerPlotting:
    """Tests for FileLogger figure and image logging."""
    
    def test_log_figure_saves_matplotlib_plot(self, tmp_path: Path):
        """Test that log_figure saves a matplotlib figure to disk."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        
        # Act
        logger.log_figure(fig, "test_plot.png")
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
        
        plt.close(fig)
    
    def test_log_figure_with_kwargs(self, tmp_path: Path):
        """Test that log_figure passes kwargs to matplotlib savefig."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        # Act
        logger.log_figure(fig, "high_dpi.png", dpi=200, bbox_inches="tight")
        logger.close()
        
        # Assert
        with Image.open(tmp_path / "high_dpi.png") as img:
            # Higher DPI should result in larger image
            assert img.size[0] >= 800  # 4 inches * 200 DPI
        
        plt.close(fig)
    
    def test_log_image_pil_artifact_mode(self, tmp_path: Path):
        """Test logging PIL Image in artifact mode."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        pil_image = Image.new("RGB", (100, 50), color="red")
        
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
    
    def test_log_image_numpy_artifact_mode(self, tmp_path: Path):
        """Test logging numpy array in artifact mode."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        numpy_image = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        
        # Act
        logger.log_image(numpy_image, artifact_file="numpy_test.png")
        logger.close()
        
        # Assert
        with Image.open(tmp_path / "numpy_test.png") as img:
            assert img.size == (120, 80)  # PIL uses (width, height)
    
    def test_log_image_time_stepped_mode(self, tmp_path: Path):
        """Test logging image in time-stepped mode with key and step."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        numpy_image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        
        # Act
        logger.log_image(numpy_image, key="training_viz", step=42)
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
        logger.log_image(unsupported_image, artifact_file="bad.png")
        logger.close()
        
        # Assert
        assert not (tmp_path / "bad.png").exists()
        
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "not supported for file saving" in log_content
    
    def test_log_image_missing_parameters_fails(self, tmp_path: Path):
        """Test that log_image without proper parameters logs error."""
        # Arrange
        logger = FileLogger(str(tmp_path / "test.log"))
        numpy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Act - no artifact_file and incomplete key+step
        logger.log_image(numpy_image, key="incomplete")  # missing step
        logger.close()
        
        # Assert
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Image logging failed" in log_content


class TestPrintLoggerPlotting:
    """Tests for PrintLogger figure and image logging."""
    
    def test_log_figure_prints_info(self, capsys):
        """Test that log_figure prints appropriate message."""
        # Arrange
        logger = PrintLogger()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        # Act
        logger.log_figure(fig, "test.png")
        logger.close()
        
        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Figure NOT saved - would be: test.png" in captured.out
        
        plt.close(fig)
    
    def test_log_image_artifact_mode_prints_info(self, capsys):
        """Test log_image in artifact mode prints correct message."""
        # Arrange
        logger = PrintLogger()
        numpy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Act
        logger.log_image(numpy_image, artifact_file="test.png")
        logger.close()
        
        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Image NOT saved - would be artifact: test.png" in captured.out
    
    def test_log_image_time_stepped_mode_prints_info(self, capsys):
        """Test log_image in time-stepped mode prints correct message."""
        # Arrange
        logger = PrintLogger()
        pil_image = Image.new("RGB", (10, 10))
        
        # Act
        logger.log_image(pil_image, key="loss_viz", step=100)
        logger.close()
        
        # Assert
        captured = capsys.readouterr()
        assert "[PrintLogger] Image NOT saved - would be key: loss_viz, step: 100" in captured.out


class TestMLFlowLoggerPlotting:
    """Tests for MLFlowLogger figure and image logging."""
    
    @patch('mlflow.MlflowClient')
    def test_log_figure_calls_client_method(self, mock_client_class):
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
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        # Act
        logger.log_figure(fig, "test.png", dpi=150)
        logger.close()
        
        # Assert
        mock_client.log_figure.assert_called_once_with(
            "run_456", fig, "test.png", dpi=150
        )
        
        plt.close(fig)
    
    @patch('mlflow.MlflowClient')
    def test_log_image_artifact_mode_calls_client(self, mock_client_class):
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
        numpy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Act
        logger.log_image(numpy_image, artifact_file="image.png")
        logger.close()
        
        # Assert
        mock_client.log_image.assert_called_once_with(
            "run_456", numpy_image, artifact_file="image.png", key=None, step=None
        )
    
    @patch('mlflow.MlflowClient')
    def test_log_image_time_stepped_mode_calls_client(self, mock_client_class):
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
        pil_image = Image.new("RGB", (20, 20))
        
        # Act
        logger.log_image(pil_image, key="training", step=50, timestamp=1234567890)
        logger.close()
        
        # Assert
        mock_client.log_image.assert_called_once_with(
            "run_456", pil_image, artifact_file=None, key="training", 
            step=50, timestamp=1234567890
        )