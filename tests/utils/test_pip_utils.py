from pathlib import Path

import pytest

from simplexity.utils.pip_utils import (
    create_requirements_file,
    get_minimal_requirements,
    get_python_version,
)


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


def test_create_requirements_file(tmp_path: Path):
    """Test create_requirements_file function."""
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text('[project]\nrequires-python = "3.12"')
    requirements_path = create_requirements_file(pyproject_toml)
    assert requirements_path == str(tmp_path / "requirements.txt")
    assert Path(requirements_path).exists()


def test_create_requirements_file_no_file(tmp_path: Path):
    """Test create_requirements_file function."""
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found at"):
        create_requirements_file("this_file_does_not_exist.toml")


def test_get_minimal_requirements(tmp_path: Path):
    """Test get_minimal_requirements function."""
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


def test_get_minimal_requirements_no_file(tmp_path: Path):
    """Test get_minimal_requirements function."""
    with pytest.raises(FileNotFoundError, match="requirements.txt not found. Run setup_mlflow_uv.py first."):
        get_minimal_requirements("this_file_does_not_exist.txt")
