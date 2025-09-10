import fcntl
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.figure
import mlflow
import numpy
import PIL.Image
import plotly.graph_objects
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from simplexity.logging.logger import Logger


class FileLogger(Logger):
    """Logs to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f"Failed to create directory for logging: {e}") from e

    def _log_to_file(self, message: str) -> None:
        """Thread-safe logging to file with file locking."""
        try:
            with open(self.file_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                print(message, file=f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            # If logging fails, we can't log the error, so raise it
            raise RuntimeError(f"Failed to write to log file: {e}") from e

    def _handle_operation_error(self, operation: str, error: Exception) -> None:
        """Common error handling pattern for file operations."""
        if isinstance(error, PermissionError):
            self._log_to_file(f"Failed to {operation} - permission denied: {error}")
        elif isinstance(error, FileNotFoundError):
            self._log_to_file(f"Failed to {operation} - file not found: {error}")
        elif isinstance(error, OSError):
            self._log_to_file(f"Failed to {operation} - OS error: {error}")
        elif isinstance(error, TypeError | ValueError):
            self._log_to_file(f"Failed to {operation} - serialization error: {error}")
        else:
            self._log_to_file(f"Failed to {operation} - unexpected error: {error}")

    def _check_disk_space(self, required_size: int, path: Path) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = os.statvfs(path.parent)
            available_bytes = stat.f_bavail * stat.f_frsize
            return available_bytes > required_size
        except Exception:
            # If we can't check, assume there's space (better than failing)
            return True

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to the file."""
        _config = OmegaConf.to_container(config, resolve=resolve)
        self._log_to_file(f"Config: {_config}")

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the file."""
        self._log_to_file(f"Metrics at step {step}: {metric_dict}")

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the file."""
        self._log_to_file(f"Params: {param_dict}")

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the file."""
        self._log_to_file(f"Tags: {tag_dict}")

    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Save figure to file system."""
        figure_path = Path(self.file_path).parent / artifact_file
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different figure types
        if isinstance(figure, matplotlib.figure.Figure):
            figure.savefig(str(figure_path), **kwargs)
        elif isinstance(figure, plotly.graph_objects.Figure):
            if str(figure_path).endswith(".html"):
                figure.write_html(str(figure_path), **kwargs)
            else:
                figure.write_image(str(figure_path), **kwargs)
        else:
            raise ValueError(f"Unsupported figure type: {type(figure)}")

        self._log_to_file(f"Figure saved: {figure_path}")

    def _save_image_to_path(
        self, image: numpy.ndarray | PIL.Image.Image | mlflow.Image, image_path: Path, **kwargs
    ) -> bool:
        """Save image to specified path. Returns True if successful, False otherwise."""
        try:
            if isinstance(image, PIL.Image.Image):
                image.save(str(image_path), **kwargs)
            elif isinstance(image, numpy.ndarray):
                Image.fromarray(image).save(str(image_path), **kwargs)
            elif isinstance(image, mlflow.Image):
                # MLflow Image objects need special handling - convert to PIL first
                pil_image = image.to_pil()
                pil_image.save(str(image_path), **kwargs)
            else:
                # Unsupported image type
                self._log_to_file(f"Image type {type(image).__name__} not supported for file saving")
                return False
            return True
        except Exception as e:
            # Log any save errors
            self._handle_operation_error("save image", e)
            return False

    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Save image to file system."""
        # Parameter validation - ensure we have either artifact_file or (key + step)
        if not artifact_file and not (key and step is not None):
            self._log_to_file("Image logging failed - need either artifact_file or (key + step)")
            return

        if artifact_file:
            # Artifact mode
            image_path = Path(self.file_path).parent / artifact_file
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if self._save_image_to_path(image, image_path, **kwargs):
                self._log_to_file(f"Image saved: {image_path}")
        else:
            # Time-stepped mode (we know key and step are valid due to validation above)
            filename = f"{key}_step_{step}.png"
            image_path = Path(self.file_path).parent / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if self._save_image_to_path(image, image_path, **kwargs):
                self._log_to_file(f"Time-stepped image saved: {image_path}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Copy artifact to the log directory with disk space validation and explicit permissions."""
        source_path = Path(local_path)
        if not source_path.exists():
            self._log_to_file(f"Artifact logging failed - file not found: {local_path}")
            return

        # Check disk space before copying
        try:
            source_size = (
                source_path.stat().st_size
                if source_path.is_file()
                else sum(f.stat().st_size for f in source_path.rglob("*") if f.is_file())
            )
        except Exception:
            source_size = 0  # If we can't determine size, proceed anyway

        # Determine destination path
        log_dir = Path(self.file_path).parent.resolve()

        if artifact_path:
            # Validate artifact_path to prevent directory traversal
            dest_path = log_dir / artifact_path
            try:
                dest_path = dest_path.resolve()
                # Ensure the resolved path is still within the log directory
                dest_path.relative_to(log_dir)
            except (OSError, ValueError):
                self._log_to_file(f"Artifact logging failed - invalid path: {artifact_path}")
                return
        else:
            dest_path = log_dir / source_path.name

        # Check disk space (with 10% buffer)
        if not self._check_disk_space(int(source_size * 1.1), dest_path):
            self._log_to_file(f"Artifact logging failed - insufficient disk space for {source_size} bytes")
            return

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if source_path.is_file():
                shutil.copy2(local_path, dest_path)
                # Set explicit permissions (readable by owner and group)
                os.chmod(dest_path, 0o644)
            else:
                shutil.copytree(local_path, dest_path, dirs_exist_ok=True)
                # Set permissions on directory and contents
                os.chmod(dest_path, 0o755)
                for root, dirs, files in os.walk(dest_path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o755)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o644)

            self._log_to_file(f"Artifact copied: {local_path} -> {dest_path}")
        except Exception as e:
            self._handle_operation_error("copy artifact", e)

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Save JSON data as an artifact to the log directory with explicit permissions."""
        # Validate artifact_name to prevent directory traversal
        log_dir = Path(self.file_path).parent.resolve()
        json_path = log_dir / artifact_name

        try:
            json_path = json_path.resolve()
            # Ensure the resolved path is still within the log directory
            json_path.relative_to(log_dir)
        except (OSError, ValueError):
            self._log_to_file(f"JSON artifact logging failed - invalid path: {artifact_name}")
            return

        json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            # Set explicit permissions (readable by owner and group)
            os.chmod(json_path, 0o644)
            self._log_to_file(f"JSON artifact saved: {json_path}")
        except Exception as e:
            self._handle_operation_error("save JSON artifact", e)

    def close(self) -> None:
        """Close the logger."""
        pass
