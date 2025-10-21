import platform
import sys
import tempfile
from pathlib import Path

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
