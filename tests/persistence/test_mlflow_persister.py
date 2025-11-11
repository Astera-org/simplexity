"""Integration-style tests for MLFlowPersister with a local MLflow backend."""

from __future__ import annotations

from pathlib import Path

import chex
import equinox as eqx
import jax
import pytest
import torch
import torch.nn as nn
import yaml

from simplexity.persistence.mlflow_persister import MLFlowPersister


def get_pytorch_model(seed: int) -> nn.Linear:
    """Build a small deterministic PyTorch model for serialization tests."""
    torch.manual_seed(seed)
    return nn.Linear(in_features=4, out_features=2)


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


def pytorch_models_equal(model1: nn.Module, model2: nn.Module) -> bool:
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
                "_target_": "eqx.nn.Linear",
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
    with open(config_path, "w") as f:
        yaml.dump(get_hydra_config_for_model(0), f)

    loaded = persister.load_model(step=0)
    chex.assert_trees_all_equal(loaded, original)


def test_mlflow_persister_cleanup(tmp_path: Path):
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="cleanup",
        run_name="cleanup-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    def run_status():
        client = persister.client
        run_id = persister.run_id
        run = client.get_run(run_id)
        return run.info.status

    assert run_status() == "RUNNING"

    model = get_model(0)
    persister.save_weights(model, step=0)
    local_persister = persister._get_local_persister(model)
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
    with open(config_path, "w") as f:
        yaml.dump(get_hydra_config_for_pytorch_model(), f)

    loaded = persister.load_model(step=0)
    assert pytorch_models_equal(loaded, original)  # type: ignore[arg-type]


def test_mlflow_persister_pytorch_cleanup(tmp_path: Path):
    """Test PyTorch model cleanup with MLflow persister."""
    artifact_dir = tmp_path / "mlruns"
    artifact_dir.mkdir()

    persister = MLFlowPersister(
        experiment_name="pytorch-cleanup",
        run_name="pytorch-cleanup-run",
        tracking_uri=artifact_dir.as_uri(),
        artifact_path="models",
    )

    def run_status():
        client = persister.client
        run_id = persister.run_id
        run = client.get_run(run_id)
        return run.info.status

    assert run_status() == "RUNNING"

    model = get_pytorch_model(0)
    persister.save_weights(model, step=0)
    local_persister = persister._get_local_persister(model)
    assert local_persister.directory.exists()

    persister.cleanup()
    assert run_status() == "FINISHED"
    assert not local_persister.directory.exists()
