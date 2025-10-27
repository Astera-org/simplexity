"""Utilities for integrating uv and pyproject.toml with MLflow model logging."""

import re
import subprocess
from pathlib import Path

MINIMAL_DEPS = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "mlflow",
    "cloudpickle",
]

CONDA_YAML_CONTENT = """name: simplexity-env
channels:
  - conda-forge
  - defaults
dependencies:
  - {python_dependency}
  - pip
  - pip:
    - -r requirements.txt
"""


def get_python_version(pyproject_path: str | Path = "pyproject.toml") -> str:
    """Get the Python version from pyproject.toml using regex."""
    pyproject_path = Path(pyproject_path)
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    python_version_match = re.search(
        r'^\s*(?:requires-)?python\s*=\s*["\']([^"\']+)["\']', pyproject_text, re.MULTILINE
    )
    if python_version_match is None:
        raise ValueError("Python version not found in pyproject.toml")
    pyproject_python_version = python_version_match.group(1).strip()
    if not pyproject_python_version:
        return "python"
    if pyproject_python_version[0] in {">", "<", "=", "!", "~"}:
        return f"python{pyproject_python_version}"
    return f"python=={pyproject_python_version}"


def create_requirements_file(pyproject_path: str | Path = "pyproject.toml") -> str:
    """Generate requirements.txt from pyproject.toml using uv."""
    pyproject_path = Path(pyproject_path)
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    requirements_path = pyproject_path.parent / "requirements.txt"
    try:
        subprocess.run(
            ["uv", "pip", "compile", str(pyproject_path), "--output-file", str(requirements_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return str(requirements_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate requirements.txt: {e.stderr}") from e


def get_minimal_requirements(requirements_path: Path | str = "requirements.txt") -> str:
    """Create a minimal requirements file with only essential dependencies."""
    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        raise FileNotFoundError("requirements.txt not found. Run setup_mlflow_uv.py first.")

    minimal_requirements_lines = [
        "# Minimal requirements for MLflow model serving",
        "# Generated from requirements.txt\n",
    ]
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                package_name = re.split(r"[>=<]", line)[0]
                if package_name in MINIMAL_DEPS:
                    minimal_requirements_lines.append(line)
    return "\n".join(minimal_requirements_lines) + "\n"


def create_minimal_requirements_file(requirements_path: Path | str = "requirements.txt") -> str:
    """Create a minimal requirements file with only essential dependencies."""
    minimal_requirements = get_minimal_requirements(requirements_path)
    requirements_path = Path(requirements_path)
    minimal_path = requirements_path.parent / "requirements_minimal.txt"
    with minimal_path.open("w") as f:
        f.write(minimal_requirements)
    return str(minimal_path)


def fix_dependency_mismatches(requirements_path: Path | str = "requirements.txt"):
    """Fix dependency mismatches between current environment and requirements.txt."""
    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        raise FileNotFoundError("requirements.txt not found. Run setup_mlflow_uv.py first.")
    requirements_path = str(requirements_path)
    subprocess.run(["uv", "pip", "install", "-r", requirements_path], check=True)


def create_conda_yaml_file(pyproject_path: str | Path = "pyproject.toml") -> str:
    """Create a conda.yaml file from pyproject.toml for MLflow."""
    pyproject_path = Path(pyproject_path)
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    python_dependency = get_python_version(pyproject_path)
    conda_yaml_path = pyproject_path.parent / "conda.yaml"
    with conda_yaml_path.open("w") as f:
        f.write(CONDA_YAML_CONTENT.format(python_dependency=python_dependency))
    return str(conda_yaml_path)
