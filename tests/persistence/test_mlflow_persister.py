"""Integration-style tests for MLFlowPersister with a local MLflow backend."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import chex
import equinox as eqx
import jax
import mlflow
import pytest
import torch
import yaml
from mlflow import MlflowClient
from mlflow.models import get_model_info, infer_signature
from torch.nn import Linear, Module

from simplexity.persistence.mlflow_persister import MLFlowPersister


def get_pytorch_model(seed: int) -> Linear:
    """Build a small deterministic PyTorch model for serialization tests."""
    torch.manual_seed(seed)
    return Linear(in_features=4, out_features=2)


def get_hydra_config_for_pytorch_model() -> dict:
    """Get a Hydra config dict for the given PyTorch model."""
    return {
        "predictive_model": {
            "instance": {
                "_target_": "torch.nn.Linear",
                "in_features": 4,
                "out_features": 2,
            }
        }
    }


def pytorch_models_equal(model1: Module, model2: Module) -> bool:
    """Check if two PyTorch models have identical parameters."""
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    if set(params1.keys()) != set(params2.keys()):
        return False

    return all(torch.allclose(params1[name], params2[name]) for name in params1)


def get_model(seed: int) -> eqx.Module:
    """Build a small deterministic model for serialization tests."""
    return eqx.nn.Linear(in_features=4, out_features=2, key=jax.random.key(seed))


def get_hydra_config_for_model(seed: int) -> dict:
    """Get a Hydra config dict for the given model."""
    return {
        "predictive_model": {
            "instance": {
                "_target_": "equinox.nn.Linear",
                "in_features": 4,
                "out_features": 2,
                "key": {"_target_": "jax.random.key", "seed": seed},
            }
        }
    }


def test_mlflow_persister_round_trip(tmp_path: Path) -> None:
    """Model weights saved via MLflow can be restored back into a new instance."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="round-trip",
        run_name="round-trip-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    original = get_model(0)
    persister.save_weights(original, step=0)

    # MLflow stores artifacts in experiment_id/run_id/artifacts/artifact_path/step/
    experiment_id = persister.experiment_id
    run_id = persister.run_id
    remote_model_path = artifact_dir / experiment_id / run_id / "artifacts" / "models" / "0" / "model.eqx"
    assert remote_model_path.exists()

    updated = get_model(1)
    loaded = persister.load_weights(updated, step=0)

    chex.assert_trees_all_equal(loaded, original)


def test_mlflow_persister_round_trip_from_config(tmp_path: Path) -> None:
    """Model weights saved via MLflow can be restored back into a new instance via the config."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="round-trip",
        run_name="round-trip-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    original = get_model(0)
    persister.save_weights(original, step=0)

    # MLflow stores artifacts in experiment_id/run_id/artifacts/artifact_path/step/
    experiment_id = persister.experiment_id
    run_id = persister.run_id
    remote_model_path = artifact_dir / experiment_id / run_id / "artifacts" / "models" / "0" / "model.eqx"
    assert remote_model_path.exists()

    # New function expects a config to live at experiment_id/run_id/artifacts/config_path
    config_path = artifact_dir / experiment_id / run_id / "artifacts" / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(get_hydra_config_for_model(0), f)

    loaded = persister.load_model(step=0)
    chex.assert_trees_all_equal(loaded, original)


def test_mlflow_persister_cleanup(tmp_path: Path) -> None:
    """Test MLflow persister cleanup."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="cleanup",
        run_name="cleanup-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    def run_status() -> str:
        """Get the status of the run."""
        client = persister.client
        run_id = persister.run_id
        run = client.get_run(run_id)
        return run.info.status

    assert run_status() == "RUNNING"

    model = get_model(0)
    persister.save_weights(model, step=0)
    local_persister = persister.get_local_persister(model)
    assert local_persister.directory.exists()

    persister.cleanup()
    assert run_status() == "FINISHED"
    assert not local_persister.directory.exists()


def test_mlflow_persister_pytorch_round_trip(tmp_path: Path) -> None:
    """PyTorch model weights saved via MLflow can be restored back into a new instance."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="pytorch-round-trip",
        run_name="pytorch-round-trip-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    original = get_pytorch_model(0)
    persister.save_weights(original, step=0)

    # MLflow stores artifacts in experiment_id/run_id/artifacts/artifact_path/step/
    experiment_id = persister.experiment_id
    run_id = persister.run_id
    remote_model_path = artifact_dir / experiment_id / run_id / "artifacts" / "models" / "0" / "model.pt"
    assert remote_model_path.exists()

    updated = get_pytorch_model(1)
    loaded = persister.load_weights(updated, step=0)

    # Type assertion since we know loaded is a PyTorch model
    assert pytorch_models_equal(loaded, original)  # type: ignore[arg-type]


def test_mlflow_persister_pytorch_round_trip_from_config(tmp_path: Path) -> None:
    """PyTorch model weights saved via MLflow can be restored back into a new instance via the config."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="pytorch-round-trip",
        run_name="pytorch-round-trip-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    original = get_pytorch_model(0)
    persister.save_weights(original, step=0)

    # MLflow stores artifacts in experiment_id/run_id/artifacts/artifact_path/step/
    experiment_id = persister.experiment_id
    run_id = persister.run_id
    remote_model_path = artifact_dir / experiment_id / run_id / "artifacts" / "models" / "0" / "model.pt"
    assert remote_model_path.exists()

    # New function expects a config to live at experiment_id/run_id/artifacts/config_path
    config_path = artifact_dir / experiment_id / run_id / "artifacts" / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(get_hydra_config_for_pytorch_model(), f)

    loaded = persister.load_model(step=0)
    assert pytorch_models_equal(loaded, original)  # type: ignore[arg-type]


def test_mlflow_persister_pytorch_cleanup(tmp_path: Path) -> None:
    """Test PyTorch model cleanup with MLflow persister."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="pytorch-cleanup",
        run_name="pytorch-cleanup-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    def run_status() -> str:
        """Get the status of the run."""
        client = persister.client
        run_id = persister.run_id
        run = client.get_run(run_id)
        return run.info.status

    assert run_status() == "RUNNING"

    model = get_pytorch_model(0)
    persister.save_weights(model, step=0)
    local_persister = persister.get_local_persister(model)
    assert local_persister.directory.exists()

    persister.cleanup()
    assert run_status() == "FINISHED"
    assert not local_persister.directory.exists()


REQUIREMENTS_CONTENT = "torch==2.0.0\nmlflow==2.0.0\nnumpy==1.20.0\n"


@pytest.fixture
def mock_create_requirements_file(tmp_path: Path) -> Generator[str, None, None]:
    """Mock the create_requirements_file function."""

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(REQUIREMENTS_CONTENT, encoding="utf-8")

    with patch("simplexity.persistence.mlflow_persister.create_requirements_file") as mock_create:
        mock_create.return_value = str(requirements_path)
        yield mock_create


@pytest.mark.usefixtures("mock_create_requirements_file")
def test_mlflow_persister_save_model_to_registry(tmp_path: Path) -> None:
    """Test saving a PyTorch model to the MLflow model registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    tracking_uri = artifact_dir.as_uri()
    registry_uri = artifact_dir.as_uri()

    persister = MLFlowPersister(
        experiment_name="registry-save",
        run_name="registry-save-run",
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    model = get_pytorch_model(0)
    registered_model_name = "test_model"

    persister.save_model_to_registry(model, registered_model_name)

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    model_versions = client.search_model_versions(filter_string=f"name='{registered_model_name}'", max_results=10)
    assert len(model_versions) == 1
    assert model_versions[0].run_id == persister.run_id
    assert model_versions[0].version == 1

    models_meta_path = artifact_dir / persister.artifact_path / registered_model_name / "version-1" / "meta.yaml"
    assert models_meta_path.exists()
    with open(models_meta_path, encoding="utf-8") as f:
        models_meta = yaml.load(f, Loader=yaml.FullLoader)
    assert models_meta["name"] == registered_model_name
    assert models_meta["run_id"] == persister.run_id
    artifact_uri = models_meta["storage_location"]
    artifact_path = Path(artifact_uri.replace("file://", ""))
    assert artifact_path.exists()
    assert artifact_path.is_dir()
    requirements_path = artifact_path / "requirements.txt"
    assert requirements_path.exists()
    assert requirements_path.is_file()
    with open(requirements_path, encoding="utf-8") as f:
        requirements_content = f.read()
    assert requirements_content == REQUIREMENTS_CONTENT.rstrip()

    persister.cleanup()


def test_mlflow_persister_save_model_to_registry_with_no_requirements(tmp_path: Path) -> None:
    """Test saving a PyTorch model to the MLflow model registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    tracking_uri = artifact_dir.as_uri()
    registry_uri = artifact_dir.as_uri()

    persister = MLFlowPersister(
        experiment_name="registry-save",
        run_name="registry-save-run",
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    model = get_pytorch_model(0)
    registered_model_name = "test_model"

    with patch("simplexity.persistence.mlflow_persister.create_requirements_file") as mock_create:
        mock_create.side_effect = FileNotFoundError(f"pyproject.toml not found at {tmp_path / 'pyproject.toml'}")
        persister.save_model_to_registry(model, registered_model_name)

    # Verify the model was registered using an independent MLflow client
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    model_versions = client.search_model_versions(filter_string=f"name='{registered_model_name}'", max_results=10)
    assert len(model_versions) == 1
    assert model_versions[0].version == 1
    assert model_versions[0].current_stage == "None"

    persister.cleanup()


@pytest.mark.usefixtures("mock_create_requirements_file")
def test_mlflow_persister_save_model_to_registry_with_model_inputs(tmp_path: Path) -> None:
    """Test saving a PyTorch model to registry with model inputs for automatic signature inference."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-save-inputs",
        run_name="registry-save-inputs-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    model = get_pytorch_model(0)
    registered_model_name = "test_model_inputs"

    # Create sample input - signature should be inferred automatically
    sample_input = torch.randn(2, 4)

    persister.save_model_to_registry(model, registered_model_name, model_inputs=sample_input)

    # Verify that the registered model has a signature
    # Set tracking and registry URIs so get_model_info uses the same URIs as the persister
    tracking_uri = artifact_dir.as_uri()
    registry_uri = artifact_dir.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    model_uri = f"models:/{registered_model_name}/1"
    model_info = get_model_info(model_uri)
    assert model_info.signature is not None, "Registered model should have a signature when model_inputs is provided"

    persister.cleanup()


def test_mlflow_persister_save_model_to_registry_non_pytorch(tmp_path: Path) -> None:
    """Test saving a non-PyTorch model to the MLflow model registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-save-non-pytorch",
        run_name="registry-save-non-pytorch-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    model = get_model(0)
    registered_model_name = "test_non_pytorch_model"

    with pytest.raises(
        ValueError,
        match=r"Model must be a PyTorch model \(torch\.nn\.Module\), got <class '.+'?>",
    ):
        persister.save_model_to_registry(model, registered_model_name)

    persister.cleanup()


@pytest.mark.usefixtures("mock_create_requirements_file")
def test_mlflow_persister_save_model_to_registry_with_signature(tmp_path: Path) -> None:
    """Test saving a PyTorch model to the MLflow model registry with a signature."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-save-signature",
        run_name="registry-save-signature-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    model = get_pytorch_model(0)
    registered_model_name = "test_model_signature"
    sample_input = torch.randn(2, 4)
    signature_data = {"some_key": "some_value", "some_other_key": "some_other_value"}
    signature = infer_signature(signature_data)

    with patch("simplexity.persistence.mlflow_persister.SIMPLEXITY_LOGGER.warning") as mock_warning:
        persister.save_model_to_registry(model, registered_model_name, model_inputs=sample_input, signature=signature)
        mock_warning.assert_called_once_with("Signature provided in kwargs, ignoring inferred signature")

    # Verify that the registered model has a signature
    # Set tracking and registry URIs so get_model_info uses the same URIs as the persister
    tracking_uri = artifact_dir.as_uri()
    registry_uri = artifact_dir.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    model_uri = f"models:/{registered_model_name}/1"
    model_info = get_model_info(model_uri)
    assert model_info.signature == signature
    persister.cleanup()


def test_mlflow_persister_load_model_from_registry(tmp_path: Path) -> None:
    """Test loading a PyTorch model from the MLflow model registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-load",
        run_name="registry-load-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    original = get_pytorch_model(0)
    registered_model_name = "test_load_model"

    # Save model to registry
    persister.save_model_to_registry(original, registered_model_name)

    # Load model from registry by version
    loaded = persister.load_model_from_registry(registered_model_name, version="1")
    assert pytorch_models_equal(loaded, original)

    # Load latest version (should be the same)
    loaded_latest = persister.load_model_from_registry(registered_model_name)
    assert pytorch_models_equal(loaded_latest, original)

    persister.cleanup()


def test_mlflow_persister_load_model_from_registry_multiple_versions(tmp_path: Path) -> None:
    """Test loading different versions of a model from the registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-multi-version",
        run_name="registry-multi-version-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    # Save first version
    model_v1 = get_pytorch_model(0)
    registered_model_name = "test_multi_version"
    persister.save_model_to_registry(model_v1, registered_model_name)

    # Create a new persister for the second version (new run)
    persister2 = MLFlowPersister(
        experiment_name="registry-multi-version",
        run_name="registry-multi-version-run-2",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    # Save second version
    model_v2 = get_pytorch_model(1)
    persister2.save_model_to_registry(model_v2, registered_model_name)

    # Load version 1
    loaded_v1 = persister.load_model_from_registry(registered_model_name, version="1")
    assert pytorch_models_equal(loaded_v1, model_v1)

    # Load version 2
    loaded_v2 = persister.load_model_from_registry(registered_model_name, version="2")
    assert pytorch_models_equal(loaded_v2, model_v2)

    # Load latest (should be version 2)
    loaded_latest = persister.load_model_from_registry(registered_model_name)
    assert pytorch_models_equal(loaded_latest, model_v2)

    persister.cleanup()
    persister2.cleanup()


def test_mlflow_persister_list_model_versions(tmp_path: Path) -> None:
    """Test listing model versions from the registry."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-list",
        run_name="registry-list-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    registered_model_name = "test_list_model"

    # Initially, no versions
    versions = persister.list_model_versions(registered_model_name)
    assert len(versions) == 0

    # Save first version
    model_v1 = get_pytorch_model(0)
    persister.save_model_to_registry(model_v1, registered_model_name)

    versions = persister.list_model_versions(registered_model_name)
    assert len(versions) == 1
    assert versions[0]["version"] == "1"
    assert "creation_timestamp" in versions[0]
    assert "last_updated_timestamp" in versions[0]

    # Save second version
    persister2 = MLFlowPersister(
        experiment_name="registry-list",
        run_name="registry-list-run-2",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )
    model_v2 = get_pytorch_model(1)
    persister2.save_model_to_registry(model_v2, registered_model_name)

    versions = persister.list_model_versions(registered_model_name)
    assert len(versions) == 2
    version_numbers = {v["version"] for v in versions}
    assert version_numbers == {"1", "2"}

    persister.cleanup()
    persister2.cleanup()


def test_mlflow_persister_load_model_from_registry_invalid_version(tmp_path: Path) -> None:
    """Test that loading a non-existent version raises an error."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-load-error",
        run_name="registry-load-error-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    registered_model_name = "test_invalid_version"

    # Try to load a model that doesn't exist
    with pytest.raises(RuntimeError):
        persister.load_model_from_registry(registered_model_name, version="999")

    persister.cleanup()


def test_mlflow_persister_load_model_from_registry_both_version_and_stage(tmp_path: Path) -> None:
    """Test that specifying both version and stage raises an error."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="registry-load-error-2",
        run_name="registry-load-error-2-run",
        tracking_uri=artifact_dir.as_uri(),
        registry_uri=artifact_dir.as_uri(),
    )

    registered_model_name = "test_both_params"

    with pytest.raises(ValueError, match="Cannot specify both version and stage. Use one or the other."):
        persister.load_model_from_registry(registered_model_name, version="1", stage="Production")

    persister.cleanup()
