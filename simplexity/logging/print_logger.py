from collections.abc import Mapping
from pprint import pprint
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf
import matplotlib.figure
import numpy
import PIL.Image
import mlflow
import plotly.graph_objects

from simplexity.logging.logger import Logger


class PrintLogger(Logger):
    """Logs to the console."""

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to the console."""
        _config = OmegaConf.to_container(config, resolve=resolve)
        pprint(f"Config: {_config}")

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the console."""
        pprint(f"Metrics at step {step}: {metric_dict}")

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the console."""
        pprint(f"Params: {param_dict}")

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the console."""
        pprint(f"Tags: {tag_dict}")

    def log_figure(
        self, 
        figure: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure], 
        artifact_file: str, 
        **kwargs,
    ) -> None:
        """Log figure info to the console (no actual figure saved)."""
        print(f"[PrintLogger] Figure NOT saved - would be: {artifact_file} (type: {type(figure).__name__})")

    def log_image(
        self, 
        image: Union[numpy.ndarray, PIL.Image.Image, mlflow.Image], 
        artifact_file: Union[str, None] = None, 
        key: Union[str, None] = None, 
        step: Union[int, None] = None, 
        **kwargs,
    ) -> None:
        """Log image info to the console (no actual image saved)."""
        if artifact_file:
            print(f"[PrintLogger] Image NOT saved - would be artifact: {artifact_file} (type: {type(image).__name__})")
        else:
            print(f"[PrintLogger] Image NOT saved - would be key: {key}, step: {step} (type: {type(image).__name__})")

    def close(self) -> None:
        """Close the logger."""
        pass
