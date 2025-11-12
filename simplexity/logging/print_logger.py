"""PrintLogger class for logging to the console."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from collections.abc import Mapping
from pprint import pprint
from typing import Any

import matplotlib.figure
import mlflow
import numpy
import PIL.Image
import plotly.graph_objects
from omegaconf import DictConfig, OmegaConf

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
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Log figure info to the console (no actual figure saved)."""
        print(f"[PrintLogger] Figure NOT saved - would be: {artifact_file} (type: {type(figure).__name__})")

    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log image info to the console (no actual image saved)."""
        # Parameter validation - ensure we have either artifact_file or (key + step)
        if not artifact_file and not (key and step is not None):
            print("[PrintLogger] Image logging failed - need either artifact_file or (key + step)")
            return

        if artifact_file:
            print(f"[PrintLogger] Image NOT saved - would be artifact: {artifact_file} (type: {type(image).__name__})")
        else:
            print(f"[PrintLogger] Image NOT saved - would be key: {key}, step: {step} (type: {type(image).__name__})")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Print artifact info to the console (no actual artifact logged)."""
        dest_name = artifact_path if artifact_path else f"<filename from {local_path}>"
        print(f"[PrintLogger] Artifact NOT logged - would copy: {local_path} -> {dest_name}")

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Print JSON artifact info to the console (no actual artifact saved)."""
        data_type = "dict" if isinstance(data, dict) else "list"
        data_size = len(data)
        print(f"[PrintLogger] JSON artifact NOT saved - would be: {artifact_name} ({data_type} with {data_size} items)")

    def close(self) -> None:
        """Close the logger."""
        pass
