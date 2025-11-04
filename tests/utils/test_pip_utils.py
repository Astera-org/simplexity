from pathlib import Path

import pytest

from simplexity.utils.pip_utils import get_python_version


def test_get_python_version_with_specific_version(tmp_path: Path):
    """Test get_python_version function."""
    specific_version_toml = tmp_path / "specific_version.toml"
    specific_version_toml.write_text('[project]\nrequires-python = "3.12"')
    assert get_python_version(specific_version_toml) == "python==3.12"


def test_get_python_version_with_range_version(tmp_path: Path):
    """Test get_python_version function."""
    range_version_toml = tmp_path / "range_version.toml"
    range_version_toml.write_text('[project]\nrequires-python = ">=3.12"')
    assert get_python_version(range_version_toml) == "python>=3.12"


def test_get_python_version_no_python(tmp_path: Path):
    """Test get_python_version function."""
    no_python_toml = tmp_path / "no_python.toml"
    no_python_toml.write_text("[project]")
    with pytest.raises(ValueError, match="Python version not found in pyproject.toml"):
        get_python_version(no_python_toml)


def test_get_python_version_no_file():
    """Test get_python_version function."""
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found at"):
        get_python_version("this_file_does_not_exist.toml")
