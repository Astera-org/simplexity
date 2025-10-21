import logging
import platform
import sys
import tempfile
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

from simplexity.logging.logger import Logger
from simplexity.utils.git import get_git_info


def log_git_info(logger: Logger) -> None:
    """Log git information for reproducibility.

    Logs git information for the main repository where training is running.
    """
    tags = {f"git.main.{k}": v for k, v in get_git_info().items()}
    if tags:
        logger.log_tags(tags)


def log_environment_artifacts(logger: Logger) -> None:
    """Log environment configuration files as MLflow artifacts for reproducibility.

    Logs dependency lockfile, project configuration, and system information
    to help reproduce the exact environment used for training.
    """
    environment_objects = ["uv.lock", "pyproject.toml"]
    for obj in environment_objects:
        if Path(obj).exists():
            logger.log_artifact(str(obj), "environment")


def log_system_info(logger: Logger) -> None:
    """Generate and log system information as an artifact."""
    with tempfile.TemporaryDirectory() as temp_dir:
        info_path = Path(temp_dir) / "system_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Architecture: {platform.architecture()}\n")
            f.write(f"Machine: {platform.machine()}\n")
            f.write(f"Processor: {platform.processor()}\n")

        logger.log_artifact(str(info_path), "environment")


def log_hydra_artifacts(logger: Logger) -> None:
    """Log Hydra artifacts for reproducibility."""
    try:
        hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
    except Exception:
        return
    hydra_artifacts = ["config.yaml", "hydra.yaml", "overrides.yaml"]
    for artifact in hydra_artifacts:
        path = hydra_dir / artifact
        if path.exists():
            try:
                logger.log_artifact(str(path), artifact_path=".hydra")
            except Exception as e:
                logging.warning("Failed to log Hydra artifact %s: %s", path, e)


def log_source_script(logger: Logger) -> None:
    """Log the source script for reproducibility."""
    try:
        # Try to get the original working directory from Hydra, fallback to current directory
        try:
            repo_root = Path(get_original_cwd())
        except Exception:
            # If Hydra is not initialized, use current working directory
            repo_root = Path.cwd()

        script_path = repo_root / __file__  # TODO: replace with actual script path
        if script_path.exists():
            logger.log_artifact(str(script_path), artifact_path="source")
    except Exception as e:
        logging.warning("Failed to log source script: %s", e)
