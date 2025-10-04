"""Demonstrate saving and loading a PyTorch model with the MLflow workspace registry."""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from mlflow.entities.model_registry import ModelVersion
from omegaconf import MISSING

from simplexity.utils.mlflow_utils import resolve_registry_uri

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - script guard
    raise SystemExit(
        "PyTorch is required for this demo. Install it with `pip install torch` "
        "or add the `pytorch` extra when installing this project."
    ) from exc


WORKSPACE_REGISTRY_URI = "databricks"


class TinyClassifier(nn.Module):
    """A tiny classifier for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass."""
        return self.model(x)


@dataclass
class DemoConfig:
    """Configuration for the MLflow workspace registry demo."""

    experiment: str = "WorkspaceRegistryDemo"
    run_name: str | None = None
    registered_model_name: str = MISSING
    tracking_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))
    registry_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_REGISTRY_URI", WORKSPACE_REGISTRY_URI))
    artifact_path: str = "pytorch-model"
    poll_interval: float = 2.0
    poll_timeout: float = 300.0
    databricks_host: str | None = field(default_factory=lambda: os.getenv("DATABRICKS_HOST"))
    allow_workspace_fallback: bool = True


CONFIG_NAME = "mlflow_workspace_registry_demo"
LEGACY_CONFIG_NAME = "mlflow_unity_catalog_demo"

config_store = ConfigStore.instance()
config_store.store(name=CONFIG_NAME, node=DemoConfig)
config_store.store(name=LEGACY_CONFIG_NAME, node=DemoConfig)


def ensure_experiment(client: mlflow.MlflowClient, name: str) -> str:
    """Ensure an experiment exists."""
    experiment = client.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    return client.create_experiment(name)


def await_model_version_ready(
    client: mlflow.MlflowClient,
    model_name: str,
    version: str,
    poll_interval: float,
    poll_timeout: float,
) -> ModelVersion:
    """Wait for a model version to be ready."""
    deadline = time.monotonic() + poll_timeout
    while True:
        current = client.get_model_version(name=model_name, version=version)
        if current.status == "READY":
            return current
        if current.status == "FAILED":
            raise RuntimeError(f"Model version {model_name}/{version} failed to register: {current.status_message}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Model version {model_name}/{version} did not become READY within {poll_timeout}s")
        time.sleep(poll_interval)


def search_model_version_for_run(
    client: mlflow.MlflowClient,
    model_name: str,
    run_id: str,
) -> ModelVersion:
    """Search for a model version for a run."""
    versions = client.search_model_versions(f"name = '{model_name}' and run_id = '{run_id}'")
    if not versions:
        raise RuntimeError(
            "No model versions were created for this run. Ensure the run has permission to register a model."
        )
    # MLflow returns the newest model version first for this query.
    return versions[0]


def build_databricks_urls(
    host: str | None,
    experiment_id: str,
    run_id: str,
    model_name: str,
    model_version: str,
) -> tuple[str | None, str | None]:
    """Build Databricks URLs for a model version."""
    if not host:
        return None, None
    base = host.rstrip("/")
    encoded_name = urllib.parse.quote(model_name, safe="")
    run_url = f"{base}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    model_url = f"{base}/#mlflow/models/{encoded_name}/versions/{model_version}"
    return run_url, model_url


def run_demo(config: DemoConfig) -> None:
    """Run the MLflow workspace registry demo."""
    resolved_registry_uri = resolve_registry_uri(
        config.tracking_uri,
        config.registry_uri,
        allow_workspace_fallback=config.allow_workspace_fallback,
    )
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    if resolved_registry_uri:
        mlflow.set_registry_uri(resolved_registry_uri)

    client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri(), registry_uri=mlflow.get_registry_uri())
    experiment_id = ensure_experiment(client, config.experiment)

    torch.manual_seed(7)
    model = TinyClassifier()
    sample_input = torch.randn(4, 4)

    run_id: str = ""  # Initialize to avoid "possibly unbound" error
    model_version: ModelVersion | None = None  # Initialize to avoid "possibly unbound" error

    with mlflow.start_run(experiment_id=experiment_id, run_name=config.run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params({"demo": "workspace_registry", "framework": "pytorch", "layers": len(list(model.modules()))})

        # First log the model without registering it
        mlflow.pytorch.log_model(  # type: ignore[attr-defined]
            model,
            artifact_path=config.artifact_path,
        )

        # Then register the model separately
        try:
            client.create_registered_model(config.registered_model_name)
            print(f"Created registered model: {config.registered_model_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Registered model {config.registered_model_name} already exists")
            else:
                raise

        # Create model version using the model URI from the logged model
        model_uri = f"runs:/{run_id}/{config.artifact_path}"
        model_version = client.create_model_version(
            name=config.registered_model_name,
            source=model_uri,
            run_id=run_id,
            description="Demo model from workspace registry",
        )
        print(f"Created model version: {model_version.version}")

        predictions = model(sample_input).detach()
        mlflow.log_artifact(
            _dump_tensor(predictions, "predictions.txt"),
            artifact_path="artifacts",
        )

    # Wait for model version to be ready
    if model_version is None:
        raise RuntimeError("Failed to create model version")
    ready_version = await_model_version_ready(
        client,
        config.registered_model_name,
        model_version.version,
        config.poll_interval,
        config.poll_timeout,
    )

    model_uri = f"models:/{config.registered_model_name}/{ready_version.version}"
    loaded_model = mlflow.pytorch.load_model(model_uri)  # type: ignore[attr-defined]
    restored_model = TinyClassifier()
    restored_model.load_state_dict(loaded_model.state_dict())

    verification_input = torch.randn(2, 4)
    original_output = model(verification_input)
    restored_output = restored_model(verification_input)
    if not torch.allclose(original_output, restored_output, atol=1e-5):
        raise RuntimeError("Loaded weights differ from the original model outputs.")

    run_url, model_url = build_databricks_urls(
        config.databricks_host,
        experiment_id,
        run_id,
        config.registered_model_name,
        ready_version.version,
    )

    info_lines = [
        "MLflow workspace registry demo complete!",
        f"Run ID: {run_id}",
        f"Model URI: {model_uri}",
        f"Model version status: {ready_version.status}",
    ]
    if run_url:
        info_lines.append(f"Run UI: {run_url}")
    if model_url:
        info_lines.append(f"Model UI: {model_url}")
    print("\n".join(info_lines))


def _dump_tensor(tensor: torch.Tensor, filename: str) -> str:
    """Dump a tensor to a file."""
    path = os.path.join(_ensure_temp_dir(), filename)
    with open(path, "w", encoding="utf-8") as handle:
        for row in tensor.tolist():
            handle.write(",".join(f"{value:.6f}" for value in row))
            handle.write("\n")
    return path


_TEMP_DIR: str | None = None


def _ensure_temp_dir() -> str:
    """Ensure a temporary directory exists."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        import tempfile

        _TEMP_DIR = tempfile.mkdtemp(prefix="mlflow-workspace-demo-")
    return _TEMP_DIR


def _cleanup_temp_dir() -> None:
    """Cleanup the temporary directory."""
    global _TEMP_DIR
    if _TEMP_DIR and os.path.isdir(_TEMP_DIR):
        import shutil

        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    _TEMP_DIR = None


def _register_atexit() -> None:
    """Register an atexit handler to cleanup the temporary directory."""
    import atexit

    atexit.register(_cleanup_temp_dir)


_register_atexit()


@hydra.main(version_base="1.2", config_name=CONFIG_NAME)
def main(config: DemoConfig) -> None:
    """Main entry point for the MLflow workspace registry demo."""
    try:
        run_demo(config)
    except (RuntimeError, TimeoutError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
