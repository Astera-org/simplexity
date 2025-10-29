"""MLflow-backed model persistence utilities."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from simplexity.persistence.local_persister import LocalPersister
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.predictive_models.types import ModelFramework, get_model_framework
from simplexity.utils.mlflow_utils import get_experiment_id, get_run_id, maybe_terminate_run, resolve_registry_uri

if TYPE_CHECKING:
    from mlflow import MlflowClient


def _build_local_persister(model_framework: ModelFramework, artifact_dir: Path) -> LocalPersister:
    if model_framework == ModelFramework.Equinox:
        from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister

        return LocalEquinoxPersister(directory=artifact_dir)
    if model_framework == ModelFramework.Penzai:
        from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister

        return LocalPenzaiPersister(directory=artifact_dir)
    if model_framework == ModelFramework.Pytorch:
        from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister

        return LocalPytorchPersister(directory=artifact_dir)


def _clear_subdirectory(subdirectory: Path) -> None:
    if subdirectory.exists():
        shutil.rmtree(subdirectory)
    subdirectory.parent.mkdir(parents=True, exist_ok=True)


class MLFlowPersister(ModelPersister):
    """Persist model checkpoints as MLflow artifacts, optionally reusing an existing run."""

    _client: MlflowClient
    _experiment_id: str
    _run_id: str
    _artifact_path: str
    _registered_model_name: str | None
    _temp_dir: tempfile.TemporaryDirectory
    _artifact_dir: Path
    _local_persisters: dict[ModelFramework, LocalPersister]

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        artifact_path: str = "models",
        registered_model_name: str | None = None,
    ):
        """Create a persister from an MLflow experiment."""
        resolved_registry_uri = resolve_registry_uri(
            registry_uri=registry_uri,
            tracking_uri=tracking_uri,
            downgrade_unity_catalog=downgrade_unity_catalog,
        )
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=resolved_registry_uri)
        self._experiment_id = get_experiment_id(experiment_name=experiment_name, client=self.client)
        self._run_id = get_run_id(experiment_id=self.experiment_id, run_name=run_name, client=self.client)
        self._artifact_path = artifact_path.strip().strip("/")
        self._registered_model_name = registered_model_name
        self._temp_dir = tempfile.TemporaryDirectory()

        # Local staging directories mirror the remote artifact layout for round-tripping.
        self._artifact_dir = (
            Path(self._temp_dir.name) / self._artifact_path if self._artifact_path else Path(self._temp_dir.name)
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._local_persisters = {}

    @property
    def client(self) -> mlflow.MlflowClient:
        """Expose underlying MLflow client for integrations."""
        return self._client

    @property
    def experiment_id(self) -> str:
        """Expose active MLflow experiment identifier."""
        return self._experiment_id

    @property
    def run_id(self) -> str:
        """Expose active MLflow run identifier."""
        return self._run_id

    @property
    def tracking_uri(self) -> str | None:
        """Return the tracking URI associated with this persister."""
        return self.client.tracking_uri

    @property
    def registry_uri(self) -> str | None:
        """Return the model registry URI associated with this persister."""
        return self.client._registry_uri

    @property
    def artifact_path(self) -> str:
        """Return the artifact path associated with this persister."""
        return self._artifact_path

    @property
    def registered_model_name(self) -> str | None:
        """Return the registered model name associated with this persister."""
        return self._registered_model_name

    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Serialize weights locally and upload them as MLflow artifacts."""
        step_dir = self._artifact_dir / str(step)
        _clear_subdirectory(step_dir)
        local_persister = self._get_local_persister(model)
        local_persister.save_weights(model, step)
        artifact_path = self._remote_step_path(step)
        try:
            self.client.log_artifacts(self.run_id, str(step_dir), artifact_path=artifact_path)
        except Exception as exc:  # pragma: no cover - exercised via mocks
            raise RuntimeError(f"Failed to log model artifacts to MLflow at step {step}") from exc
        self._maybe_register_model(artifact_path)

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Download MLflow artifacts and restore them into the provided model."""
        step_dir = self._artifact_dir / str(step)
        _clear_subdirectory(step_dir)
        artifact_path = self._remote_step_path(step)
        try:
            downloaded_path = Path(
                self.client.download_artifacts(
                    self.run_id,
                    artifact_path,
                    dst_path=str(self._temp_dir.name),
                )
            )
        except Exception as exc:  # pragma: no cover - exercised via mocks
            raise RuntimeError(f"Failed to download model artifacts from MLflow at step {step}") from exc

        if not downloaded_path.exists():
            raise RuntimeError(f"MLflow artifact for step {step} was not found after download")

        local_persister = self._get_local_persister(model)
        return local_persister.load_weights(model, step)

    def cleanup(self) -> None:
        """Remove temporary resources and optionally end the MLflow run."""
        for persister in self._local_persisters.values():
            persister.cleanup()
        self._temp_dir.cleanup()
        maybe_terminate_run(self.run_id, client=self.client)

    def _get_local_persister(self, model: PredictiveModel) -> LocalPersister:
        model_framework = get_model_framework(model)
        if model_framework not in self._local_persisters:
            self._local_persisters[model_framework] = _build_local_persister(model_framework, self._artifact_dir)
        return self._local_persisters[model_framework]

    def _remote_step_path(self, step: int) -> str:
        parts: list[str] = []
        if self.artifact_path:
            parts.append(self.artifact_path)
        parts.append(str(step))
        return "/".join(parts)

    def _maybe_register_model(self, artifact_path: str) -> None:
        if not self.registered_model_name:
            return

        # Check if model exists, create if it doesn't
        matches = self.client.search_registered_models(
            filter_string=f"name = '{self.registered_model_name}'",
            max_results=1,
        )
        if not matches:
            with contextlib.suppress(Exception):
                self.client.create_registered_model(self.registered_model_name)

        source = f"runs:/{self.run_id}/{artifact_path}"
        with contextlib.suppress(Exception):
            # Surface registration failures as warnings while allowing training to proceed.
            self.client.create_model_version(
                name=self.registered_model_name,
                source=source,
                run_id=self.run_id,
            )
