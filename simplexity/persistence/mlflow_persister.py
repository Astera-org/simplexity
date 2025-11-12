"""MLflow-backed model persistence utilities."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.types import ModelFramework, get_model_framework
from simplexity.run_management.structured_configs import InstanceConfig
from simplexity.utils.config_utils import typed_instantiate
from simplexity.utils.mlflow_utils import get_experiment_id, get_run_id, maybe_terminate_run, resolve_registry_uri


def _build_local_persister(model_framework: ModelFramework, artifact_dir: Path) -> LocalPersister:
    if model_framework == ModelFramework.EQUINOX:
        from simplexity.persistence.local_equinox_persister import (  # pylint: disable=import-outside-toplevel
            LocalEquinoxPersister,
        )

        directory = artifact_dir / "equinox"
        return LocalEquinoxPersister(directory=directory)
    if model_framework == ModelFramework.PENZAI:
        from simplexity.persistence.local_penzai_persister import (  # pylint: disable=import-outside-toplevel
            LocalPenzaiPersister,
        )

        directory = artifact_dir / "penzai"
        return LocalPenzaiPersister(directory=directory)
    if model_framework == ModelFramework.PYTORCH:
        from simplexity.persistence.local_pytorch_persister import (  # pylint: disable=import-outside-toplevel
            LocalPytorchPersister,
        )

        directory = artifact_dir / "pytorch"
        return LocalPytorchPersister(directory=directory)

    raise ValueError(f"Unsupported model framework: {model_framework}")


def _clear_subdirectory(subdirectory: Path) -> None:
    if subdirectory.exists():
        shutil.rmtree(subdirectory)
    subdirectory.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class MLFlowPersisterInstanceConfig(InstanceConfig):
    """Configuration for the MLflow persister."""

    experiment_name: str
    run_name: str
    tracking_uri: str | None = None
    registry_uri: str | None = None
    downgrade_unity_catalog: bool = True
    artifact_path: str = "models"
    config_path: str = "config.yaml"


class MLFlowPersister:  # pylint: disable=too-many-instance-attributes
    """Persist model checkpoints as MLflow artifacts, optionally reusing an existing run."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        artifact_path: str = "models",
        config_path: str = "config.yaml",
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
        self._temp_dir = tempfile.TemporaryDirectory()
        self._artifact_dir = (
            Path(self._temp_dir.name) / self._artifact_path if self._artifact_path else Path(self._temp_dir.name)
        )
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = config_path
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
        return self.client._registry_uri  # pylint: disable=protected-access

    def save_weights(self, model: Any, step: int = 0) -> None:
        """Serialize weights locally and upload them as MLflow artifacts."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        local_persister.save_weights(model, step)
        framework_dir = step_dir.parent
        self.client.log_artifacts(self.run_id, str(framework_dir), artifact_path=self._artifact_path)

    def load_weights(self, model: Any, step: int = 0) -> Any:
        """Download MLflow artifacts and restore them into the provided model."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        artifact_path = f"{self._artifact_path}/{step}"
        downloaded_path = self.client.download_artifacts(
            self.run_id,
            artifact_path,
            dst_path=str(step_dir.parent),
        )
        if not Path(downloaded_path).exists():
            raise RuntimeError(f"MLflow artifact for step {step} was not found after download")
        return local_persister.load_weights(model, step)

    def load_model(self, step: int = 0) -> Any:
        """Load a model from a specified MLflow run and step."""
        config_path = self._config_path

        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded_config_path = self.client.download_artifacts(
                self.run_id,
                config_path,
                dst_path=str(temp_dir),
            )
            run_config = OmegaConf.load(downloaded_config_path)

        instance: DictConfig = OmegaConf.select(run_config, "predictive_model.instance", throw_on_missing=True)
        target: str = OmegaConf.select(run_config, "predictive_model.instance._target_", throw_on_missing=True)
        model = typed_instantiate(instance, target)

        return self.load_weights(model, step)

    def cleanup(self) -> None:
        """Remove temporary resources and optionally end the MLflow run."""
        for persister in self._local_persisters.values():
            persister.cleanup()
        self._temp_dir.cleanup()
        maybe_terminate_run(run_id=self.run_id, client=self.client)

    def get_local_persister(self, model: Any) -> LocalPersister:
        """Get the local persister for the given model."""
        model_framework = get_model_framework(model)
        if model_framework not in self._local_persisters:
            self._local_persisters[model_framework] = _build_local_persister(model_framework, self._artifact_dir)
        return self._local_persisters[model_framework]
