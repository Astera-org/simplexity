"""MLFlow tracker."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import dotenv
import matplotlib.figure
import mlflow
import mlflow.pytorch as mlflow_pytorch
import numpy
import PIL.Image
import plotly.graph_objects
import torch
from mlflow.entities import Metric, Param, RunTag
from mlflow.models.model import ModelInfo
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf

from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.predictive_models.types import ModelFramework, get_model_framework
from simplexity.structured_configs.tracking import MLFlowTrackerInstanceConfig
from simplexity.tracking.model_persistence.local_model_persister import (
    LocalModelPersister,
)
from simplexity.tracking.tracker import RunTracker
from simplexity.tracking.utils import build_local_persister
from simplexity.utils.mlflow_utils import (
    get_experiment,
    get_run,
    maybe_terminate_run,
    resolve_registry_uri,
    set_mlflow_uris,
)
from simplexity.utils.pip_utils import create_requirements_file


def _clear_subdirectory(subdirectory: Path) -> None:
    if subdirectory.exists():
        shutil.rmtree(subdirectory)
    subdirectory.parent.mkdir(parents=True, exist_ok=True)


dotenv.load_dotenv()


class MLFlowTracker(RunTracker):  # pylint: disable=too-many-instance-attributes
    """Tracks runs to MLflow."""

    def __init__(
        self,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool | None = None,
        model_dir: str = "models",
        config_path: str = "config.yaml",
    ):
        """Initialize MLflow tracker."""
        self._downgrade_unity_catalog = downgrade_unity_catalog if downgrade_unity_catalog is not None else True
        resolved_registry_uri = resolve_registry_uri(
            registry_uri=registry_uri,
            tracking_uri=tracking_uri,
            downgrade_unity_catalog=downgrade_unity_catalog,
        )
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=resolved_registry_uri)
        experiment = get_experiment(experiment_id=experiment_id, experiment_name=experiment_name, client=self.client)
        assert experiment is not None
        self._experiment_id = experiment.experiment_id
        self._experiment_name = experiment.name
        run = get_run(run_id=run_id, run_name=run_name, experiment_id=self.experiment_id, client=self.client)
        assert run is not None
        self._run_id = run.info.run_id
        self._run_name = run.info.run_name

        # Model persistence setup
        self._model_dir = model_dir.strip().strip("/")
        self._temp_dir = tempfile.TemporaryDirectory()
        self._model_path = Path(self._temp_dir.name) / self._model_dir if self._model_dir else Path(self._temp_dir.name)
        self._model_path.mkdir(parents=True, exist_ok=True)
        self._config_path = config_path
        self._local_persisters: dict[ModelFramework, LocalModelPersister] = {}

    @property
    def client(self) -> mlflow.MlflowClient:
        """Expose underlying MLflow client for integrations."""
        return self._client

    @property
    def experiment_name(self) -> str:
        """Expose active MLflow experiment name."""
        return self._experiment_name

    @property
    def experiment_id(self) -> str:
        """Expose active MLflow experiment identifier."""
        return self._experiment_id

    @property
    def run_name(self) -> str | None:
        """Expose active MLflow run name."""
        return self._run_name

    @property
    def run_id(self) -> str:
        """Expose active MLflow run identifier."""
        return self._run_id

    @property
    def tracking_uri(self) -> str | None:
        """Return the tracking URI associated with this tracker."""
        return self.client.tracking_uri

    @property
    def registry_uri(self) -> str | None:
        """Return the model registry URI associated with this tracker."""
        return self.client._registry_uri  # pylint: disable=protected-access

    @property
    def model_dir(self) -> str:
        """Return the artifact path associated with this tracker."""
        return self._model_dir

    @property
    def cfg(self) -> MLFlowTrackerInstanceConfig:
        """Return the configuration of this tracker."""
        return MLFlowTrackerInstanceConfig(
            _target_=f"simpexity.tracking.{self.__class__.__module__}.{self.__class__.__qualname__}",
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            run_name=self.run_name,
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
            downgrade_unity_catalog=self._downgrade_unity_catalog,
            model_dir=self.model_dir,
            config_path=self._config_path,
        )

    # Lifecycle

    def close(self) -> None:
        """End the MLflow run."""
        self.cleanup()

    def cleanup(self) -> None:
        """Remove temporary resources and optionally end the MLflow run."""
        for persister in self._local_persisters.values():
            persister.cleanup()
        self._temp_dir.cleanup()
        maybe_terminate_run(run_id=self.run_id, client=self.client)

    # Logging

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to MLflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            OmegaConf.save(config, config_path, resolve=resolve)
            self.client.log_artifact(self.run_id, config_path)

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

    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Log a figure to MLflow using MLflowClient.log_figure."""
        self.client.log_figure(self.run_id, figure, artifact_file, **kwargs)

    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log an image to MLflow using MLflowClient.log_image."""
        if not artifact_file and not (key and step is not None):
            raise ValueError("Must provide either artifact_file or both key and step parameters")

        self.client.log_image(self.run_id, image, artifact_file=artifact_file, key=key, step=step, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact (file or directory) to MLflow."""
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    def log_json_artifact(self, data: dict | list, artifact_name: str) -> None:
        """Log a JSON object as an artifact to MLflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, artifact_name)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.client.log_artifact(self.run_id, json_path)

    def _log_batch(self, **kwargs: Any) -> None:
        """Log arbitrary data to MLflow."""
        self.client.log_batch(self.run_id, **kwargs, synchronous=False)

    # Persistence

    def save_model(self, model: Any, step: int = 0) -> None:
        """Serialize weights locally and upload them as MLflow artifacts."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        local_persister.save_weights(model, step)
        framework_dir = step_dir.parent
        self.client.log_artifacts(self.run_id, str(framework_dir), artifact_path=self._model_dir)

    def load_model(self, model: Any, step: int = 0) -> Any:
        """Download MLflow artifacts and restore them into the provided model."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        artifact_path = f"{self._model_dir}/{step}"
        downloaded_path = self.client.download_artifacts(
            self.run_id,
            artifact_path,
            dst_path=str(step_dir.parent),
        )
        if not Path(downloaded_path).exists():
            raise RuntimeError(f"MLflow artifact for step {step} was not found after download")
        return local_persister.load_weights(model, step)

    def get_local_persister(self, model: Any) -> LocalModelPersister:
        """Get the local persister for the given model."""
        model_framework = get_model_framework(model)
        if model_framework not in self._local_persisters:
            self._local_persisters[model_framework] = build_local_persister(model_framework, self._model_path)
        return self._local_persisters[model_framework]

    # Model Registry

    def save_model_to_registry(
        self,
        model: Any,
        registered_model_name: str,
        model_inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelInfo:
        """Save a PyTorch model to the MLflow model registry."""
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Model must be a PyTorch model (torch.nn.Module), got {type(model)}")

        signature = None
        if model_inputs is not None:
            model.eval()
            with torch.no_grad():
                model_outputs: torch.Tensor = model(model_inputs)
            signature = infer_signature(
                model_input=model_inputs.detach().cpu().numpy(),
                model_output=model_outputs.detach().cpu().numpy(),
            )

        log_kwargs: dict[str, Any] = {
            "pytorch_model": model,
            "registered_model_name": registered_model_name,
        }

        if "signature" in kwargs:
            log_kwargs["signature"] = kwargs.pop("signature")
            if signature is not None:
                SIMPLEXITY_LOGGER.warning("Signature provided in kwargs, ignoring inferred signature")
        elif signature is not None:
            log_kwargs["signature"] = signature

        if "pip_requirements" not in kwargs:
            try:
                pip_requirements = create_requirements_file()
                log_kwargs["pip_requirements"] = pip_requirements
            except (FileNotFoundError, RuntimeError):
                SIMPLEXITY_LOGGER.warning("Failed to generate pip requirements file, continuing without it")

        log_kwargs.update(kwargs)

        model_info: ModelInfo | None = None
        with set_mlflow_uris(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri):
            active_run = mlflow.active_run()
            if active_run is not None and active_run.info.run_id != self.run_id:
                raise RuntimeError(
                    "Cannot save model to registry because an active MLflow run "
                    f"({active_run.info.run_id}) does not match the persister run id ({self.run_id}). "
                    "End the active run or use the same run id."
                )
            run_context = mlflow.start_run(run_id=self.run_id) if active_run is None else nullcontext()
            with run_context:
                model_info = mlflow_pytorch.log_model(**log_kwargs)
        assert model_info is not None
        return model_info

    def registered_model_uri(
        self, registered_model_name: str, version: str | None = None, stage: str | None = None
    ) -> str:
        """Get the URI for a registered model."""
        prefix = "models:"
        if version is not None and stage is not None:
            raise ValueError("Cannot specify both version and stage. Use one or the other.")
        if stage is not None:
            return f"{prefix}/{registered_model_name}/{stage}"
        if version is not None:
            return f"{prefix}/{registered_model_name}/{version}"

        model_versions = self.client.search_model_versions(
            filter_string=f"name='{registered_model_name}'", max_results=1, order_by=["version_number DESC"]
        )
        if not model_versions:
            raise RuntimeError(f"No versions found for registered model '{registered_model_name}'")
        latest_version = model_versions[0].version
        return f"{prefix}/{registered_model_name}/{latest_version}"

    def load_model_from_registry(
        self,
        registered_model_name: str,
        version: str | None = None,
        stage: str | None = None,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> Any:
        """Load a PyTorch model from the MLflow model registry."""
        model_uri = self.registered_model_uri(registered_model_name, version, stage)
        with set_mlflow_uris(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri):
            return mlflow_pytorch.load_model(model_uri)
