"""MLflow-backed model persistence utilities."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch as mlflow_pytorch
import torch
from mlflow.models.model import ModelInfo
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf

from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.types import ModelFramework, get_model_framework
from simplexity.utils.config_utils import typed_instantiate
from simplexity.utils.mlflow_utils import (
    get_experiment_id,
    get_run_id,
    maybe_terminate_run,
    resolve_registry_uri,
    set_mlflow_uris,
)
from simplexity.utils.pip_utils import create_requirements_file

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


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


class MLFlowPersister:  # pylint: disable=too-many-instance-attributes
    """Persist model checkpoints as MLflow artifacts, optionally reusing an existing run."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        downgrade_unity_catalog: bool = True,
        model_dir: str = "models",
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
        self._model_dir = model_dir.strip().strip("/")
        self._temp_dir = tempfile.TemporaryDirectory()
        self._model_path = Path(self._temp_dir.name) / self._model_dir if self._model_dir else Path(self._temp_dir.name)
        self._model_path.mkdir(parents=True, exist_ok=True)
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

    @property
    def model_dir(self) -> str:
        """Return the artifact path associated with this persister."""
        return self._model_dir

    def save_weights(self, model: Any, step: int = 0) -> None:
        """Serialize weights locally and upload them as MLflow artifacts."""
        local_persister = self.get_local_persister(model)
        step_dir = local_persister.directory / str(step)
        _clear_subdirectory(step_dir)
        local_persister.save_weights(model, step)
        framework_dir = step_dir.parent
        self.client.log_artifacts(self.run_id, str(framework_dir), artifact_path=self._model_dir)

    def load_weights(self, model: Any, step: int = 0) -> Any:
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
            self._local_persisters[model_framework] = _build_local_persister(model_framework, self._model_path)
        return self._local_persisters[model_framework]

    def save_model_to_registry(
        self,
        model: Any,
        registered_model_name: str,
        model_inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelInfo:
        """Save a PyTorch model to the MLflow model registry.

        Args:
            model: The PyTorch model to save. Must be a torch.nn.Module instance.
            registered_model_name: The name to register the model under in the registry.
            model_inputs: Optional model inputs (torch.Tensor) to use for inferring the model signature.
                         If provided, the signature will be automatically inferred.
            **kwargs: Additional keyword arguments passed to mlflow.pytorch.log_model.
                     Can include 'signature' or 'pip_requirements' to override defaults.

        Raises:
            ValueError: If the model is not a PyTorch model.
        """
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
        """Get the URI for a registered model.

        Args:
            registered_model_name: The name of the registered model.
            version: Optional specific version to load (e.g., "1", "2"). If None, loads the latest version.
            stage: Optional stage to load from (e.g., "Production", "Staging", "Archived").
            If provided, takes precedence over version.
        """
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
    ) -> Any:
        """Load a PyTorch model from the MLflow model registry.

        Args:
            registered_model_name: The name of the registered model.
            version: Optional specific version to load (e.g., "1", "2"). If None, loads the latest version.
            stage: Optional stage to load from (e.g., "Production", "Staging", "Archived").
                   If provided, takes precedence over version.

        Returns:
            The loaded PyTorch model.

        Raises:
            ValueError: If both version and stage are provided.
            RuntimeError: If the model cannot be found or loaded.
        """
        model_uri = self.registered_model_uri(registered_model_name, version, stage)
        with set_mlflow_uris(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri):
            return mlflow_pytorch.load_model(model_uri)

    def list_model_versions(
        self,
        registered_model_name: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List available versions for a registered model.

        Args:
            registered_model_name: The name of the registered model.
            max_results: Maximum number of versions to return.

        Returns:
            A list of dictionaries containing version information. Each dictionary includes:
            - version: The version number (string)
            - stage: The current stage (string)
            - status: The version status (string)
            - creation_timestamp: When the version was created (timestamp)
            - last_updated_timestamp: When the version was last updated (timestamp)
        """
        model_versions = self.client.search_model_versions(
            filter_string=f"name='{registered_model_name}'", max_results=max_results
        )

        return [
            {
                "version": mv.version,
                "stage": mv.current_stage,
                "status": mv.status,
                "creation_timestamp": mv.creation_timestamp,
                "last_updated_timestamp": mv.last_updated_timestamp,
            }
            for mv in model_versions
        ]
