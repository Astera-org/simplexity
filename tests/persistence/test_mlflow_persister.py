"""Integration-style tests for MLFlowPersister with a local MLflow backend."""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.mlflow_persister import MLFlowPersister
from simplexity.predictive_models.gru_rnn import GRURNN

# PyTorch imports with conditional availability
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
else:

    def get_pytorch_model(seed: int) -> nn.Linear:
        """Build a small deterministic PyTorch model for serialization tests."""
        torch.manual_seed(seed)
        return nn.Linear(in_features=4, out_features=2)

    def pytorch_models_equal(model1: nn.Module, model2: nn.Module) -> bool:
        """Check if two PyTorch models have identical parameters."""
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())

        if set(params1.keys()) != set(params2.keys()):
            return False

        return all(torch.allclose(params1[name], params2[name]) for name in params1)


def get_model(seed: int) -> GRURNN:
    """Build a small deterministic model for serialization tests."""
    return GRURNN(vocab_size=2, embedding_size=4, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


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


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch is not available")
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


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch is not available")
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
