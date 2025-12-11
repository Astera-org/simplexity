"""Test the file tracker."""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from omegaconf import DictConfig
from PIL import Image

from simplexity.tracking.file_tracker import FileTracker

EXPECTED_LOG = """Config: {'str_param': 'str_value', 'int_param': 1, 'float_param': 1.0, 'bool_param': True}
Config: {'str_param': 'str_value', 'int_param': 1, 'float_param': 1.0, 'bool_param': True}
Params: {'str_param': 'str_value', 'int_param': 1, 'float_param': 1.0, 'bool_param': True}
Tags: {'str_tag': 'str_value', 'int_tag': 1, 'float_tag': 1.0, 'bool_tag': True}
Metrics at step 1: {'int_metric': 1, 'float_metric': 1.0, 'jnp_metric': Array(0.1, dtype=float32, weak_type=True)}
"""

EXPECTED_LOG_WITH_INTERPOLATION = (
    "Config: {'base_value': 'hello', 'interpolated_value': 'hello_world', 'nested': {'value': 'hello_nested'}}\n"
)


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
def sample_json_data():
    """Sample JSON data for testing."""
    return {"key": "value", "number": 42, "list": [1, 2, 3]}


@pytest.fixture
def sample_list_data():
    """Sample list data for testing."""
    return [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]


def test_file_tracker(tmp_path: Path):
    """Test FileTracker initialization."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    params = {
        "str_param": "str_value",
        "int_param": 1,
        "float_param": 1.0,
        "bool_param": True,
    }
    tracker.log_config(DictConfig(params))
    tracker.log_config(DictConfig(params), resolve=True)
    tracker.log_params(params)
    tags = {
        "str_tag": "str_value",
        "int_tag": 1,
        "float_tag": 1.0,
        "bool_tag": True,
    }
    tracker.log_tags(tags)
    metrics = {
        "int_metric": 1,
        "float_metric": 1.0,
        "jnp_metric": jnp.array(0.1),
    }
    tracker.log_metrics(1, metrics)
    tracker.close()

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        assert f.read() == EXPECTED_LOG


def test_file_tracker_with_interpolation(tmp_path: Path):
    """Test that resolved config properly resolves interpolations."""
    tracker = FileTracker(str(tmp_path / "test.log"))

    # Create a config with interpolation
    config_dict = {
        "base_value": "hello",
        "interpolated_value": "${base_value}_world",
        "nested": {
            "value": "${base_value}_nested",
        },
    }

    config = DictConfig(config_dict)
    tracker.log_config(config, resolve=True)
    tracker.close()

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        assert f.read() == EXPECTED_LOG_WITH_INTERPOLATION


def test_log_artifact_copies_file(test_artifact_file, tmp_path: Path):
    """Test that log_artifact copies a file to the log directory."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_artifact(str(test_artifact_file))
    tracker.close()

    copied_file = tmp_path / "test_artifact.txt"
    assert copied_file.exists()
    assert copied_file.read_text() == "test content"

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Artifact copied:" in log_content
        assert "test_artifact.txt" in log_content


def test_log_artifact_with_custom_path(tmp_path: Path):
    """Test log_artifact with custom artifact path."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    test_file = tmp_path / "source.txt"
    test_file.write_text("content")

    tracker.log_artifact(str(test_file), "custom/path/dest.txt")
    tracker.close()

    copied_file = tmp_path / "custom" / "path" / "dest.txt"
    assert copied_file.exists()
    assert copied_file.read_text() == "content"


def test_log_artifact_directory(test_artifact_directory, tmp_path: Path):
    """Test that log_artifact can copy entire directories."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_artifact(str(test_artifact_directory), "copied_dir")
    tracker.close()

    copied_dir = tmp_path / "copied_dir"
    assert copied_dir.is_dir()
    assert (copied_dir / "file1.txt").read_text() == "content1"
    assert (copied_dir / "file2.txt").read_text() == "content2"


def test_log_json_artifact_saves_json(sample_json_data, tmp_path: Path):
    """Test that log_json_artifact saves JSON data."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_json_artifact(sample_json_data, "results.json")
    tracker.close()

    json_file = tmp_path / "results.json"
    assert json_file.exists()

    with open(json_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
        assert loaded_data == sample_json_data

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "JSON artifact saved:" in log_content
        assert "results.json" in log_content


def test_log_json_artifact_with_list(sample_list_data, tmp_path: Path):
    """Test log_json_artifact with list data."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_json_artifact(sample_list_data, "data_list.json")
    tracker.close()

    json_file = tmp_path / "data_list.json"
    assert json_file.exists()

    with open(json_file, encoding="utf-8") as f:
        loaded_data = json.load(f)
        assert loaded_data == sample_list_data


def test_log_figure_saves_matplotlib_plot(matplotlib_figure, tmp_path: Path):
    """Test that log_figure saves a matplotlib figure to disk."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_figure(matplotlib_figure, "test_plot.png")
    tracker.close()

    with Image.open(tmp_path / "test_plot.png") as img:
        assert img.size == (400, 300)

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Figure saved:" in log_content
        assert "test_plot.png" in log_content


def test_log_figure_with_kwargs(simple_matplotlib_figure, tmp_path: Path):
    """Test that log_figure passes kwargs to matplotlib savefig."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_figure(simple_matplotlib_figure, "high_dpi.png", dpi=200, bbox_inches="tight")
    tracker.close()

    with Image.open(tmp_path / "high_dpi.png") as img:
        assert img.size[0] >= 800


def test_log_image_pil_artifact_mode(pil_image, tmp_path: Path):
    """Test logging PIL Image in artifact mode."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_image(pil_image, artifact_file="pil_test.png")
    tracker.close()

    with Image.open(tmp_path / "pil_test.png") as img:
        assert img.size == (100, 50)

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Image saved:" in log_content
        assert "pil_test.png" in log_content


def test_log_image_numpy_artifact_mode(numpy_image, tmp_path: Path):
    """Test logging numpy array in artifact mode."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_image(numpy_image, artifact_file="numpy_test.png")
    tracker.close()

    with Image.open(tmp_path / "numpy_test.png") as img:
        assert img.size == (120, 80)


def test_log_image_time_stepped_mode(small_numpy_image, tmp_path: Path):
    """Test logging image in time-stepped mode with key and step."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_image(small_numpy_image, key="training_viz", step=42)
    tracker.close()

    with Image.open(tmp_path / "training_viz_step_42.png") as img:
        assert img.size == (50, 50)

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Time-stepped image saved:" in log_content
        assert "training_viz_step_42.png" in log_content


def test_log_image_unsupported_type(tmp_path: Path):
    """Test logging unsupported image type logs error."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    unsupported_image = "not an image"

    tracker.log_image(unsupported_image, artifact_file="bad.png")  # type: ignore[arg-type]
    tracker.close()

    assert not (tmp_path / "bad.png").exists()

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "not supported for file saving" in log_content


def test_log_image_missing_parameters_fails(tiny_numpy_image, tmp_path: Path):
    """Test that log_image without proper parameters logs error."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_image(tiny_numpy_image, key="incomplete")  # missing step
    tracker.close()

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Image logging failed" in log_content


def test_log_image_no_parameters_fails(tiny_numpy_image, tmp_path: Path):
    """Test that log_image with no parameters logs error."""
    tracker = FileTracker(str(tmp_path / "test.log"))
    tracker.log_image(tiny_numpy_image)  # Neither artifact_file nor key+step
    tracker.close()

    with open(tmp_path / "test.log", encoding="utf-8") as f:
        log_content = f.read()
        assert "Image logging failed - need either artifact_file or (key + step)" in log_content
