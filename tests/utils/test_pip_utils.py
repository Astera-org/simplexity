"""Tests for pip utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simplexity.utils.pip_utils import (
    create_conda_yaml_file,
    create_minimal_requirements_file,
    create_requirements_file,
    fix_dependency_mismatches,
    get_minimal_requirements,
    get_python_version,
)


def test_get_python_version_with_specific_version(tmp_path: Path):
    """Test get_python_version with specific version."""
    specific_version_toml = tmp_path / "specific_version.toml"
    specific_version_toml.write_text('[project]\nrequires-python = "3.12"')
    assert get_python_version(specific_version_toml) == "python==3.12"


def test_get_python_version_with_range_version(tmp_path: Path):
    """Test get_python_version with range version."""
    range_version_toml = tmp_path / "range_version.toml"
    range_version_toml.write_text('[project]\nrequires-python = ">=3.12"')
    assert get_python_version(range_version_toml) == "python>=3.12"


def test_get_python_version_no_python(tmp_path: Path):
    """Test get_python_version with no Python version."""
    no_python_toml = tmp_path / "no_python.toml"
    no_python_toml.write_text("[project]")
    with pytest.raises(ValueError, match="Python version not found in pyproject.toml"):
        get_python_version(no_python_toml)


def test_get_python_version_no_file():
    """Test get_python_version with no file."""
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found at"):
        get_python_version("this_file_does_not_exist.toml")


def test_create_requirements_file(tmp_path: Path):
    """Test create_requirements_file."""
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('[project]\nrequires-python = "3.12"')

    def mock_run_uv_export(*args, **_kwargs):
        output_file = None
        if args and len(args) > 0:
            cmd = args[0]
            if "--output-file" in cmd:
                idx = cmd.index("--output-file")
                if idx + 1 < len(cmd):
                    output_file = Path(cmd[idx + 1])
        if output_file:
            output_file.write_text("numpy==1.24.0\n", encoding="utf-8")
        return MagicMock(returncode=0)

    with patch("simplexity.utils.pip_utils.subprocess.run") as mock_subprocess_run:
        mock_subprocess_run.side_effect = mock_run_uv_export
        requirements_path = create_requirements_file(pyproject_toml)
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        assert call_args[0][0] == [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--output-file",
            str(requirements_path),
        ]

    assert requirements_path == str(tmp_path / "requirements.txt")
    assert Path(requirements_path).exists()


def test_create_requirements_file_no_file():
    """Test create_requirements_file with no file."""
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found at"):
        create_requirements_file("this_file_does_not_exist.toml")


def test_get_minimal_requirements(tmp_path: Path):
    """Test get_minimal_requirements."""
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("""
cloudpickle>2.0.0
equinox==0.13.0
hydra-core==1.2.0
jax==0.8.0
jupyter==2.0.0
matplotlib==3.5.0
mlflow!=2.0.0
numpy<=1.24.0
optax==0.1.0
pandas===2.0.0
penzai==0.1.0
scikit-learn~=1.2.0
torch=2.0.0
transformer-lens>=2.15.4
""")
    assert (
        get_minimal_requirements(requirements_path)
        == """# Minimal requirements for MLflow model serving
# Generated from requirements.txt

cloudpickle>2.0.0
mlflow!=2.0.0
numpy<=1.24.0
pandas===2.0.0
scikit-learn~=1.2.0
torch=2.0.0
"""
    )


def test_get_minimal_requirements_no_file():
    """Test get_minimal_requirements with no file."""
    with pytest.raises(FileNotFoundError, match="requirements.txt not found. Run setup_mlflow_uv.py first."):
        get_minimal_requirements("this_file_does_not_exist.txt")


def test_create_minimal_requirements_file(tmp_path: Path):
    """Test create_minimal_requirements_file."""
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("torch==2.0.0")
    minimal_requirements_path = create_minimal_requirements_file(requirements_path)
    assert minimal_requirements_path == str(tmp_path / "requirements_minimal.txt")
    assert Path(minimal_requirements_path).exists()
    with open(minimal_requirements_path, encoding="utf-8") as f:
        assert (
            f.read()
            == """# Minimal requirements for MLflow model serving
# Generated from requirements.txt

torch==2.0.0
"""
        )


def test_fix_dependency_mismatches_no_file():
    """Test fix_dependency_mismatches with no file."""
    with pytest.raises(FileNotFoundError, match="requirements.txt not found. Run setup_mlflow_uv.py first."):
        fix_dependency_mismatches("this_file_does_not_exist.txt")


def test_create_conda_yaml_file(tmp_path: Path):
    """Test create_conda_yaml_file."""
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('[project]\nrequires-python = "3.12"')
    conda_yaml_path = create_conda_yaml_file(pyproject_toml)
    assert conda_yaml_path == str(tmp_path / "conda.yaml")
    assert Path(conda_yaml_path).exists()
    with open(conda_yaml_path, encoding="utf-8") as f:
        assert (
            f.read()
            == """name: simplexity-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python==3.12
  - pip
  - pip:
    - -r requirements.txt
"""
        )


def test_create_conda_yaml_file_no_file():
    """Test create_conda_yaml_file with no file."""
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found at"):
        create_conda_yaml_file("this_file_does_not_exist.toml")
