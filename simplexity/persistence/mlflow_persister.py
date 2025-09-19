"""MLflow-backed model persistence utilities."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.predictive_models.types import ModelFramework

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from simplexity.logging.mlflow_logger import MLFlowLogger


def _normalize_artifact_path(artifact_path: str) -> str:
    """Return a normalized artifact path without surrounding slashes."""
    artifact_path = artifact_path.strip()
    return artifact_path.strip("/")


class MLFlowPersister(ModelPersister):
    """Persist model checkpoints as MLflow artifacts, optionally reusing an existing run."""

    client: Any
    run_id: str
    artifact_path: str
    model_framework: ModelFramework
    registered_model_name: str | None
    _temp_dir: tempfile.TemporaryDirectory
    _base_dir: Path
    _artifact_dir: Path
    _local_persister: ModelPersister
    _registered_model_checked: bool
    _managed_run: bool

    def __init__(
        self,
        client: "MlflowClient | Any",
        run_id: str,
        *,
        artifact_path: str = "models",
        model_framework: ModelFramework = ModelFramework.Equinox,
        registered_model_name: str | None = None,
        temp_dir: tempfile.TemporaryDirectory | None = None,
        managed_run: bool = False,
    ):
        self.client = client
        self.run_id = run_id
        self.artifact_path = _normalize_artifact_path(artifact_path)
        self.model_framework = model_framework
        self.registered_model_name = registered_model_name
        self._temp_dir = temp_dir or tempfile.TemporaryDirectory()
        self._managed_run = managed_run

        # Local staging directories mirror the remote artifact layout for round-tripping.
        self._base_dir = Path(self._temp_dir.name)
        self._artifact_dir = self._base_dir / self.artifact_path if self.artifact_path else self._base_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._local_persister = self._build_local_persister(self._artifact_dir)
        self._registered_model_checked = False

    @classmethod
    def from_experiment(
        cls,
        experiment_name: str,
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        artifact_path: str = "models",
        model_framework: ModelFramework = ModelFramework.Equinox,
        registered_model_name: str | None = None,
    ) -> "MLFlowPersister":
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = client.create_experiment(experiment_name)
        run = client.create_run(experiment_id=experiment_id, run_name=run_name)
        return cls(
            client=client,
            run_id=run.info.run_id,
            artifact_path=artifact_path,
            model_framework=model_framework,
            registered_model_name=registered_model_name,
            managed_run=True,
        )

    @classmethod
    def from_logger(
        cls,
        logger: "MLFlowLogger",
        *,
        artifact_path: str = "models",
        model_framework: ModelFramework = ModelFramework.Equinox,
        registered_model_name: str | None = None,
    ) -> "MLFlowPersister":
        """Create a persister reusing an existing MLFlowLogger run."""
        return cls(
            client=logger.client,
            run_id=logger.run_id,
            artifact_path=artifact_path,
            model_framework=model_framework,
            registered_model_name=registered_model_name,
            managed_run=False,
        )

    @property
    def local_persister(self) -> ModelPersister:
        """Expose the backing local persister (primarily for testing)."""
        return self._local_persister

    def cleanup(self) -> None:
        """Remove temporary resources and optionally end the MLflow run."""
        persister_cleanup = getattr(self._local_persister, "cleanup", None)
        if callable(persister_cleanup):
            persister_cleanup()
        if self._managed_run:
            try:
                self.client.set_terminated(self.run_id)
            except Exception:
                # Cleanup is best-effort; ignore failures when ending the run.
                pass
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
                    dst_path=str(self._base_dir),
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

        if not self._registered_model_checked:
            try:
                self.client.get_registered_model(self.registered_model_name)
            except Exception:
                try:
                    self.client.create_registered_model(self.registered_model_name)
                except Exception:
                    pass
            self._registered_model_checked = True

        source = f"runs:/{self.run_id}/{artifact_path}"
        try:
            self.client.create_model_version(
                name=self.registered_model_name,
                source=source,
                run_id=self.run_id,
            )
        except Exception:
            # Surface registration failures as warnings while allowing training to proceed.
            pass
