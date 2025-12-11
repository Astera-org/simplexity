"""File tracker."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import json
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

from simplexity.predictive_models.types import ModelFramework, get_model_framework
from simplexity.tracking.model_persistence.local_model_persister import (
    LocalModelPersister,
)
from simplexity.tracking.tracker import RunTracker
from simplexity.tracking.utils import build_local_persister


def _clear_subdirectory(subdirectory: Path) -> None:
    if subdirectory.exists():
        shutil.rmtree(subdirectory)
    subdirectory.parent.mkdir(parents=True, exist_ok=True)


class FileTracker(RunTracker):
    """Tracks runs to a file/directory."""

    def __init__(self, file_path: str, model_dir_name: str = "models"):
        self.file_path = Path(file_path)
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f"Failed to create directory for logging: {e}") from e

        # Model persistence
        self._model_dir = self.file_path.parent / model_dir_name
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._local_persisters: dict[ModelFramework, LocalModelPersister] = {}

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to the file."""
        with open(self.file_path, "a") as f:
            _config = OmegaConf.to_container(config, resolve=resolve)
            print(f"Config: {_config}", file=f)

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the file."""
        with open(self.file_path, "a") as f:
            print(f"Metrics at step {step}: {metric_dict}", file=f)

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the file."""
        with open(self.file_path, "a") as f:
            print(f"Params: {param_dict}", file=f)

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the file."""
        with open(self.file_path, "a") as f:
            print(f"Tags: {tag_dict}", file=f)

    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Save figure to file system."""
        figure_path = self.file_path.parent / artifact_file
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

        with open(self.file_path, "a") as f:
            print(f"Figure saved: {figure_path}", file=f)

    def _save_image_to_path(
        self, image: numpy.ndarray | PIL.Image.Image | mlflow.Image, image_path: Path, **kwargs
    ) -> bool:
        """Save image to specified path. Returns True if successful, False otherwise."""
        try:
            if isinstance(image, PIL.Image.Image):
                image.save(str(image_path), **kwargs)
            elif isinstance(image, numpy.ndarray):
                PIL.Image.fromarray(image).save(str(image_path), **kwargs)
            elif isinstance(image, mlflow.Image):
                # MLflow Image objects need special handling - convert to PIL first
                pil_image = image.to_pil()
                pil_image.save(str(image_path), **kwargs)
            else:
                # Unsupported image type
                with open(self.file_path, "a") as f:
                    print(f"Image type {type(image).__name__} not supported for file saving", file=f)
                return False
            return True
        except Exception as e:
            # Log any save errors
            with open(self.file_path, "a") as f:
                print(f"Failed to save image: {e}", file=f)
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
            with open(self.file_path, "a") as f:
                print("Image logging failed - need either artifact_file or (key + step)", file=f)
            return

        if artifact_file:
            # Artifact mode
            image_path = self.file_path.parent / artifact_file
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if self._save_image_to_path(image, image_path, **kwargs):
                with open(self.file_path, "a") as f:
                    print(f"Image saved: {image_path}", file=f)
        else:
            # Time-stepped mode
            filename = f"{key}_step_{step}.png"
            image_path = self.file_path.parent / filename
            image_path.parent.mkdir(parents=True, exist_ok=True)

            if self._save_image_to_path(image, image_path, **kwargs):
                with open(self.file_path, "a") as f:
                    print(f"Time-stepped image saved: {image_path}", file=f)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Copy artifact to the log directory."""
        source_path = Path(local_path)
        dest_path = self.file_path.parent / (artifact_path or source_path.name)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            shutil.copy2(local_path, dest_path)
        else:
            shutil.copytree(local_path, dest_path, dirs_exist_ok=True)

        with open(self.file_path, "a") as f:
            print(f"Artifact copied: {local_path} -> {dest_path}", file=f)

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Save JSON data as an artifact to the log directory."""
        json_path = self.file_path.parent / artifact_name
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        with open(self.file_path, "a") as f:
            print(f"JSON artifact saved: {json_path}", file=f)

    def cleanup(self) -> None:
        """Cleanup resources."""
        for persister in self._local_persisters.values():
            persister.cleanup()

    # Persistence

    def save_model(self, model: Any, step: int = 0) -> None:
        """Save a model to the file system."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        local_persister.save_weights(model, step)
        # Note: Local persisters already save to the model_dir which is under file_path.parent
        # So we just need to ensure the local persister is built with the right root.

    def load_model(self, model: Any, step: int = 0) -> Any:
        """Load a model from the file system."""
        local_persister = self.get_local_persister(model)
        return local_persister.load_weights(model, step)

    def get_local_persister(self, model: Any) -> LocalModelPersister:
        """Get the local persister for the given model."""
        model_framework = get_model_framework(model)
        if model_framework not in self._local_persisters:
            self._local_persisters[model_framework] = build_local_persister(model_framework, self._model_dir)
        return self._local_persisters[model_framework]

    # Model Registry (Not supported)
    def save_model_to_registry(self, model: Any, registered_model_name: str, **kwargs) -> Any:
        """Save a model to the registry (Not Supported)."""
        raise NotImplementedError("FileTracker does not support model registry.")

    def load_model_from_registry(self, registered_model_name: str, **kwargs) -> Any:
        """Load a model from the registry (Not Supported)."""
        raise NotImplementedError("FileTracker does not support model registry.")
