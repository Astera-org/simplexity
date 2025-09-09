import os
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import dotenv
import mlflow
from mlflow.entities import Metric, Param, RunTag
from omegaconf import DictConfig, OmegaConf

from simplexity.logging.logger import Logger

dotenv.load_dotenv()
_DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")


class MLFlowLogger(Logger):
    """Logs to MLflow Tracking."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow logger."""
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._client.create_experiment(experiment_name)
        run = self._client.create_run(experiment_id=experiment_id, run_name=run_name)
        self._run_id = run.info.run_id

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to MLflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            OmegaConf.save(config, config_path, resolve=resolve)
            self._client.log_artifact(self._run_id, config_path)

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to MLflow."""
        timestamp = int(time.time() * 1000)
        metrics = self._flatten_metric_dict(metric_dict, timestamp, step)
        self._log_batch(metrics=metrics)

    def _flatten_metric_dict(
        self, metric_dict: Mapping[str, Any], timestamp: int, step: int, key_prefix: str = ""
    ) -> list[Metric]:
        """Flatten a dictionary of metrics into a list of Metric entities."""
        metrics = []
        for key, value in metric_dict.items():
            key = f"{key_prefix}/{key}" if key_prefix else key
            if isinstance(value, Mapping):
                nested_metrics = self._flatten_metric_dict(value, timestamp, step, key_prefix=key)
                metrics.extend(nested_metrics)
            else:
                value = float(value)
                metric = Metric(key, value, timestamp, step)
                metrics.append(metric)
        return metrics

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to MLflow."""
        params = self._flatten_param_dict(param_dict)
        self._log_batch(params=params)

    def _flatten_param_dict(self, param_dict: Mapping[str, Any], key_prefix: str = "") -> list[Param]:
        """Flatten a dictionary of params into a list of Param entities."""
        params = []
        for key, value in param_dict.items():
            key = f"{key_prefix}.{key}" if key_prefix else key
            if isinstance(value, Mapping):
                nested_params = self._flatten_param_dict(value, key_prefix=key)
                params.extend(nested_params)
            else:
                value = str(value)
                param = Param(key, value)
                params.append(param)
        return params

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Set tags on the MLFlow."""
        tags = [RunTag(k, str(v)) for k, v in tag_dict.items()]
        self._log_batch(tags=tags)

    def log_figure(self, figure, artifact_file: str, **kwargs) -> None:
        """Log a figure to MLflow using MLflowClient.log_figure."""
        self._client.log_figure(self._run_id, figure, artifact_file, **kwargs)

    def log_image(
        self, image, artifact_file: str | None = None, key: str | None = None, step: int | None = None, **kwargs
    ) -> None:
        """Log an image to MLflow using MLflowClient.log_image."""
        self._client.log_image(self._run_id, image, artifact_file=artifact_file, key=key, step=step, **kwargs)

    def close(self):
        """End the MLflow run."""
        self._client.set_terminated(self._run_id)

    def _log_batch(self, **kwargs: Any) -> None:
        """Log arbitrary data to MLflow."""
        self._client.log_batch(self._run_id, **kwargs, synchronous=False)
