"""RunTracker protocol."""

from typing import Any, Protocol
from collections.abc import Mapping

import mlflow
import numpy
import PIL.Image
import plotly.graph_objects
import matplotlib.figure
from omegaconf import DictConfig


class RunTracker(Protocol):
    """Tracks run data (metrics, params, artifacts, models)."""

    # Lifecycle
    def close(self) -> None:
        """Close the tracker."""
        ...

    def cleanup(self) -> None:
        """Cleanup resources."""
        ...

    # Logging
    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config."""
        ...

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics."""
        ...

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params."""
        ...

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags."""
        ...

    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Log a figure."""
        ...

    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log an image."""
        ...

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact (file or directory)."""
        ...

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Log a JSON object as an artifact."""
        ...

    # Model Persistence
    def save_model(self, model: Any, step: int = 0) -> None:
        """Save a model."""
        ...

    def load_model(self, model: Any, step: int = 0) -> Any:
        """Load a model."""
        ...

    # Model Registry (Optional)
    def save_model_to_registry(self, model: Any, registered_model_name: str, **kwargs) -> Any:
        """Save a model to the registry."""
        ...

    def load_model_from_registry(self, registered_model_name: str, **kwargs) -> Any:
        """Load a model from the registry."""
        ...

    # Data Retrieval & Listing (Future Scope)
    # def list_run_data(self) -> dict[str, Any]: ...
    # def download_run_data(self, ...): ...
