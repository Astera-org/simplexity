"""Run logging utilities for logging run information."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import inspect
import platform
import sys
import tempfile
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.tracking.tracker import RunTracker
from simplexity.utils.git_utils import get_git_info


def _get_calling_file_path() -> str | None:
    """Get the file path of the script that called the decorated function."""
    # TODO: not the most robust, contains hardcoded heuristics for identifying internal modules
    try:
        # Get the current frame and walk up the stack to find the calling file
        current_frame = inspect.currentframe()
        if current_frame:
            # Walk up the stack to find the first non-built-in file that's not an internal module
            frame = current_frame.f_back
            this_module_path = Path(__file__).resolve()
            run_management_path = this_module_path.parent / "run_management.py"

            # Get the project root to identify internal modules
            project_root = this_module_path.parent.parent

            while frame:
                frame_path = Path(frame.f_code.co_filename).resolve()

                # Skip built-in modules
                if frame.f_code.co_filename.startswith("<"):
                    frame = frame.f_back
                    continue

                # Skip this module and run_management
                if frame_path in (this_module_path, run_management_path):
                    frame = frame.f_back
                    continue

                # Skip internal library files (hydra, simplexity modules, etc.)
                # Only skip if it's actually a library file, not a user script
                if (
                    "site-packages" in str(frame_path)
                    or "hydra" in str(frame_path).lower()
                    or (
                        frame_path.is_relative_to(project_root)
                        and "simplexity" in str(frame_path).lower()
                        and "simplexity/" in str(frame_path).lower()
                        and frame_path.suffix == ".py"
                    )
                ):
                    frame = frame.f_back
                    continue

                # This looks like a user script
                return str(frame_path)
    except Exception:
        # If we can't get the calling file path, return None
        pass
    return None


def log_git_info(logger: RunTracker) -> None:
    """Log git information for reproducibility.

    Logs git information for the main repository where training is running.
    """
    tags = {f"git.main.{k}": v for k, v in get_git_info().items()}
    if tags:
        logger.log_tags(tags)


def log_environment_artifacts(logger: RunTracker) -> None:
    """Log environment configuration files as MLflow artifacts for reproducibility.

    Logs dependency lockfile, project configuration, and system information
    to help reproduce the exact environment used for training.
    """
    environment_objects = ["uv.lock", "pyproject.toml"]
    for obj in environment_objects:
        if Path(obj).exists():
            logger.log_artifact(str(obj), "environment")


def log_system_info(logger: RunTracker) -> None:
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


def log_hydra_artifacts(logger: RunTracker) -> None:
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
                SIMPLEXITY_LOGGER.warning("Failed to log Hydra artifact %s: %s", path, e)


def log_source_script(logger: RunTracker) -> None:
    """Log the source script for reproducibility."""
    calling_file_path = _get_calling_file_path()
    if calling_file_path:
        logger.log_artifact(calling_file_path, artifact_path="source")
    else:
        SIMPLEXITY_LOGGER.warning("Failed to log source script")
