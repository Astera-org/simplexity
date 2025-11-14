"""Logger interface for logging to a variety of backends."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import matplotlib.figure
import mlflow
import numpy
import PIL.Image
import plotly.graph_objects
from omegaconf import DictConfig


class Logger(ABC):
    """Logs to a variety of backends."""

    @abstractmethod
    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to the logger."""

    @abstractmethod
    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the logger."""

    @abstractmethod
    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the logger."""

    @abstractmethod
    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the logger."""

    @abstractmethod
    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Log a figure to the logger."""

    @abstractmethod
    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log an image to the logger.

        Args:
            image: Image to log (numpy array, PIL Image, or mlflow Image)
            artifact_file: File path for artifact mode (e.g., "image.png")
            key: Key name for time-stepped mode (requires step parameter)
            step: Step number for time-stepped mode (requires key parameter)
            **kwargs: Additional arguments passed to the underlying save method

        Note:
            Must provide either artifact_file OR both key and step parameters.
            Providing neither or only one of key/step will result in an error.
        """

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact (file or directory) to the logger.

        Args:
            local_path: Path to the local file or directory to log
            artifact_path: Optional artifact path within the experiment run.
                          If None, uses the filename from local_path.
        """

    @abstractmethod
    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Log a JSON object as an artifact to the logger.

        Args:
            data: Dictionary or list to serialize as JSON
            artifact_name: Name for the artifact (e.g., "results.json")
        """

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
