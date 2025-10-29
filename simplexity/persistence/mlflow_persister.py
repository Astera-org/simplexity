"""MLflow-backed model persistence utilities."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.predictive_models.types import ModelFramework
from simplexity.utils.mlflow_utils import get_experiment_id, get_run_id, maybe_terminate_run, resolve_registry_uri

if TYPE_CHECKING:
    from mlflow import MlflowClient


class MLFlowPersister(ModelPersister):
    """Persist model checkpoints as MLflow artifacts, optionally reusing an existing run."""

    _client: MlflowClient
    _run_id: str
    artifact_path: str
    model_framework: ModelFramework
    registered_model_name: str | None
    _temp_dir: tempfile.TemporaryDirectory
    _artifact_dir: Path
    _local_persister: ModelPersister

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        artifact_path: str = "models",
        model_framework: ModelFramework = ModelFramework.Pytorch,
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
        self._model_framework = model_framework
        self._registered_model_name = registered_model_name
        self._temp_dir = tempfile.TemporaryDirectory()

        # Local staging directories mirror the remote artifact layout for round-tripping.
        self._artifact_dir = (
            Path(self._temp_dir.name) / self.artifact_path if self.artifact_path else Path(self._temp_dir.name)
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._local_persister = self._build_local_persister(self._artifact_dir)

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
    def local_persister(self) -> ModelPersister:
        """Expose the backing local persister (primarily for testing)."""
        return self._local_persister

    def cleanup(self) -> None:
        """Remove temporary resources and optionally end the MLflow run."""
        persister_cleanup = getattr(self._local_persister, "cleanup", None)
        if callable(persister_cleanup):
            persister_cleanup()
        maybe_terminate_run(self.run_id, client=self.client)
        self._temp_dir.cleanup()

    def save_weights(self, model: PredictiveModel, step: int = 0) -> None:
        """Serialize weights locally and upload them as MLflow artifacts."""
        self._clear_step_dir(step)
        step_dir = self._artifact_dir / str(step)
        self._local_persister.save_weights(model, step)
        artifact_path = self._remote_step_path(step)
        try:
            self.client.log_artifacts(self.run_id, str(step_dir), artifact_path=artifact_path)
        except Exception as exc:  # pragma: no cover - exercised via mocks
            raise RuntimeError(f"Failed to log model artifacts to MLflow at step {step}") from exc
        self._maybe_register_model(artifact_path)

    def load_weights(self, model: PredictiveModel, step: int = 0) -> PredictiveModel:
        """Download MLflow artifacts and restore them into the provided model."""
        self._clear_step_dir(step)
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

        return self._local_persister.load_weights(model, step)

    def _build_local_persister(self, directory: Path) -> ModelPersister:
        if self.model_framework == ModelFramework.Equinox:
            from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister

            return LocalEquinoxPersister(directory)
        if self.model_framework == ModelFramework.Penzai:
            from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister

            return LocalPenzaiPersister(directory)
        if self.model_framework == ModelFramework.Pytorch:
            from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister

            return LocalPytorchPersister(directory)
        raise ValueError(f"Unsupported model framework: {self.model_framework}")

    def _remote_step_path(self, step: int) -> str:
        parts: list[str] = []
        if self.artifact_path:
            parts.append(self.artifact_path)
        parts.append(str(step))
        return "/".join(parts)

    def _clear_step_dir(self, step: int) -> None:
        step_dir = self._artifact_dir / str(step)
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.parent.mkdir(parents=True, exist_ok=True)

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
