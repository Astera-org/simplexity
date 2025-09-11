"""Tests for artifact logging functionality across all logger implementations."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simplexity.logging.file_logger import FileLogger
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.logging.print_logger import PrintLogger


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing."""
    return {"key": "value", "number": 42, "list": [1, 2, 3]}


@pytest.fixture
def sample_list_data():
    """Sample list data for testing."""
    return [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]


@pytest.fixture
def test_artifact_file(tmp_path: Path):
    """Create a test file to use as an artifact."""
    test_file = tmp_path / "source" / "test_artifact.txt"
    test_file.parent.mkdir()
    test_file.write_text("test content")
    return test_file


@pytest.fixture
def test_artifact_directory(tmp_path: Path):
    """Create a test directory to use as an artifact."""
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    (source_dir / "file1.txt").write_text("content1")
    (source_dir / "file2.txt").write_text("content2")
    return source_dir


@pytest.fixture
def file_logger(tmp_path: Path):
    """Create FileLogger with temporary path."""
    return FileLogger(str(tmp_path / "test.log"))


@pytest.fixture
def print_logger():
    """Create PrintLogger."""
    return PrintLogger()


class TestFileLoggerArtifacts:
    """Tests for FileLogger artifact logging."""

    def test_log_artifact_copies_file(self, file_logger, test_artifact_file, tmp_path: Path):
        """Test that log_artifact copies a file to the log directory."""
        # Act
        file_logger.log_artifact(str(test_artifact_file))
        file_logger.close()

        # Assert
        copied_file = tmp_path / "test_artifact.txt"
        assert copied_file.exists()
        assert copied_file.read_text() == "test content"

        # Verify log content
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "Artifact copied:" in log_content
            assert "test_artifact.txt" in log_content

    def test_log_artifact_with_custom_path(self, file_logger, tmp_path: Path):
        """Test log_artifact with custom artifact path."""
        # Arrange
        test_file = tmp_path / "source.txt"
        test_file.write_text("content")

        # Act
        file_logger.log_artifact(str(test_file), "custom/path/dest.txt")
        file_logger.close()

        # Assert
        copied_file = tmp_path / "custom" / "path" / "dest.txt"
        assert copied_file.exists()
        assert copied_file.read_text() == "content"

    def test_log_artifact_directory(self, file_logger, test_artifact_directory, tmp_path: Path):
        """Test that log_artifact can copy entire directories."""
        # Act
        file_logger.log_artifact(str(test_artifact_directory), "copied_dir")
        file_logger.close()

        # Assert
        copied_dir = tmp_path / "copied_dir"
        assert copied_dir.is_dir()
        assert (copied_dir / "file1.txt").read_text() == "content1"
        assert (copied_dir / "file2.txt").read_text() == "content2"

    def test_log_json_artifact_saves_json(self, file_logger, sample_json_data, tmp_path: Path):
        """Test that log_json_artifact saves JSON data."""
        # Act
        file_logger.log_json_artifact(sample_json_data, "results.json")
        file_logger.close()

        # Assert
        json_file = tmp_path / "results.json"
        assert json_file.exists()

        with open(json_file) as f:
            loaded_data = json.load(f)
            assert loaded_data == sample_json_data

        # Verify log content
        with open(tmp_path / "test.log") as f:
            log_content = f.read()
            assert "JSON artifact saved:" in log_content
            assert "results.json" in log_content

    def test_log_json_artifact_with_list(self, file_logger, sample_list_data, tmp_path: Path):
        """Test log_json_artifact with list data."""
        # Act
        file_logger.log_json_artifact(sample_list_data, "data_list.json")
        file_logger.close()

        # Assert
        json_file = tmp_path / "data_list.json"
        assert json_file.exists()

        with open(json_file) as f:
            loaded_data = json.load(f)
            assert loaded_data == sample_list_data


class TestPrintLoggerArtifacts:
    """Tests for PrintLogger artifact logging."""

    def test_log_artifact_prints_info(self, print_logger, capsys):
        """Test that log_artifact prints appropriate message."""
        # Act
        print_logger.log_artifact("/path/to/file.txt")
        print_logger.close()

        # Assert
        captured = capsys.readouterr()
        expected = (
            "[PrintLogger] Artifact NOT logged - would copy: /path/to/file.txt -> <filename from /path/to/file.txt>"
        )
        assert expected in captured.out

    def test_log_artifact_with_custom_path_prints_info(self, print_logger, capsys):
        """Test log_artifact with custom path prints correct message."""
        # Act
        print_logger.log_artifact("/source.txt", "dest/path.txt")
        print_logger.close()

        # Assert
        captured = capsys.readouterr()
        expected = "[PrintLogger] Artifact NOT logged - would copy: /source.txt -> dest/path.txt"
        assert expected in captured.out

    def test_log_json_artifact_prints_info(self, print_logger, sample_json_data, capsys):
        """Test log_json_artifact prints correct message."""
        # Act
        print_logger.log_json_artifact(sample_json_data, "test.json")
        print_logger.close()

        # Assert
        captured = capsys.readouterr()
        expected = "[PrintLogger] JSON artifact NOT saved - would be: test.json (dict with 3 items)"
        assert expected in captured.out

    def test_log_json_artifact_list_prints_info(self, print_logger, sample_list_data, capsys):
        """Test log_json_artifact with list prints correct message."""
        # Act
        print_logger.log_json_artifact(sample_list_data, "list.json")
        print_logger.close()

        # Assert
        captured = capsys.readouterr()
        expected = "[PrintLogger] JSON artifact NOT saved - would be: list.json (list with 2 items)"
        assert expected in captured.out


class TestMLFlowLoggerArtifacts:
    """Tests for MLFlowLogger artifact logging."""

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

    @pytest.fixture
    def mock_mlflow_logger(self):
        """Create a mocked MLFlowLogger for testing."""
        with patch("mlflow.MlflowClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.get_experiment_by_name.return_value = None
            mock_client.create_experiment.return_value = "exp_123"
            mock_run = MagicMock()
            mock_run.info.run_id = "run_456"
            mock_client.create_run.return_value = mock_run

            logger = MLFlowLogger("test_experiment")
            yield logger, mock_client

    def test_log_artifact_calls_client(self, mock_mlflow_logger):
        """Test that log_artifact calls the MLflow client correctly."""
        # Arrange
        logger, mock_client = mock_mlflow_logger

        # Act
        logger.log_artifact("/path/to/file.txt", "artifacts/file.txt")
        logger.close()

        # Assert
        mock_client.log_artifact.assert_called_once_with("run_456", "/path/to/file.txt", "artifacts/file.txt")

    def test_log_artifact_without_artifact_path(self, mock_mlflow_logger):
        """Test log_artifact without custom artifact path."""
        # Arrange
        logger, mock_client = mock_mlflow_logger

        # Act
        logger.log_artifact("/path/to/model.pkl")
        logger.close()

        # Assert
        mock_client.log_artifact.assert_called_once_with("run_456", "/path/to/model.pkl", None)

    def test_log_json_artifact_calls_client(self, mock_mlflow_logger, sample_json_data):
        """Test that log_json_artifact creates temp file and calls client."""
        # Arrange
        logger, mock_client = mock_mlflow_logger

        # Act
        logger.log_json_artifact(sample_json_data, "metrics.json")
        logger.close()

        # Assert
        mock_client.log_artifact.assert_called_once()
        call_args = mock_client.log_artifact.call_args
        assert call_args[0][0] == "run_456"  # run_id
        assert call_args[0][1].endswith("metrics.json")  # temp file path
        # log_json_artifact calls with only 2 args (no artifact_path)
        assert len(call_args[0]) == 2
