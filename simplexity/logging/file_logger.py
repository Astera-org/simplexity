from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf
from PIL import Image
import matplotlib.figure
import mlflow
import numpy
import PIL.Image
import plotly.graph_objects

from simplexity.logging.logger import Logger


class FileLogger(Logger):
    """Logs to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f"Failed to create directory for logging: {e}") from e

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
        figure: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure],
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

        with open(self.file_path, "a") as f:
            print(f"Figure saved: {figure_path}", file=f)

    def log_image(
        self,
        image: Union[numpy.ndarray, PIL.Image.Image, mlflow.Image],
        artifact_file: Union[str, None] = None,
        key: Union[str, None] = None,
        step: Union[int, None] = None,
        **kwargs,
    ) -> None:
        """Save image to file system."""
        if artifact_file:
            image_path = Path(self.file_path).parent / artifact_file
            image_path.parent.mkdir(parents=True, exist_ok=True)
            # Handle different image types
            if isinstance(image, PIL.Image.Image):
                image.save(str(image_path), **kwargs)
            elif isinstance(image, numpy.ndarray):
                Image.fromarray(image).save(str(image_path), **kwargs)
            elif isinstance(image, mlflow.Image):
                # MLflow Image objects need special handling - convert to PIL first
                pil_image = image.to_pil()
                pil_image.save(str(image_path), **kwargs)
            else:
                # Fallback - just log that we can't save it
                with open(self.file_path, "a") as f:
                    print(f"Image type {type(image).__name__} not supported for file saving", file=f)
                return
            with open(self.file_path, "a") as f:
                print(f"Image saved: {image_path}", file=f)
        else:
            # For time-stepped logging, create a filename using key and step
            if key and step is not None:
                filename = f"{key}_step_{step}.png"
                image_path = Path(self.file_path).parent / filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                # Same image saving logic
                if isinstance(image, PIL.Image.Image):
                    image.save(str(image_path), **kwargs)
                elif isinstance(image, numpy.ndarray):
                    Image.fromarray(image).save(str(image_path), **kwargs)
                elif isinstance(image, mlflow.Image):
                    # MLflow Image objects need special handling - convert to PIL first
                    pil_image = image.to_pil()
                    pil_image.save(str(image_path), **kwargs)
                else:
                    with open(self.file_path, "a") as f:
                        print(f"Image type {type(image).__name__} not supported for file saving", file=f)
                    return
                with open(self.file_path, "a") as f:
                    print(f"Time-stepped image saved: {image_path}", file=f)
            else:
                with open(self.file_path, "a") as f:
                    print("Image logging failed - need either artifact_file or (key + step)", file=f)

    def close(self) -> None:
        """Close the logger."""
        pass
