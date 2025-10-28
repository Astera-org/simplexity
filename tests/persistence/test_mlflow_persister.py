"""Tests for MLFlowPersister behavior."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.persistence.mlflow_persister import MLFlowPersister
from simplexity.predictive_models.gru_rnn import GRURNN
from simplexity.predictive_models.types import ModelFramework

chex = pytest.importorskip("chex")
jax = pytest.importorskip("jax")


def get_model(seed: int) -> GRURNN:
    return GRURNN(vocab_size=2, embedding_size=4, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


@pytest.fixture
def mlflow_client_mock(tmp_path: Path) -> tuple[MagicMock, Path]:
    """Create an MlflowClient mock that simulates artifact storage."""
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    client = MagicMock()

    def log_artifacts(run_id: str, local_dir: str, artifact_path: str | None = None):
        assert run_id == "run_123"
        destination = remote_root if artifact_path is None else remote_root / artifact_path
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, destination)

    def download_artifacts(run_id: str, path: str, dst_path: str | None = None) -> str:
        assert run_id == "run_123"
        source = remote_root / path
        if not source.exists():
            raise FileNotFoundError(path)
        base_dir = Path(dst_path) if dst_path else tmp_path / "downloads"
        destination = base_dir / path
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return str(destination)

    client.log_artifacts.side_effect = log_artifacts
    client.download_artifacts.side_effect = download_artifacts
    return client, remote_root


def test_mlflow_persister_round_trip(tmp_path: Path, mlflow_client_mock: tuple[MagicMock, Path]):
    """Model weights saved via MLFLow can be restored back into a new instance."""
    client, remote_root = mlflow_client_mock

    persister = MLFlowPersister(
        client=client,
        run_id="run_123",
        artifact_path="models",
        model_framework=ModelFramework.Equinox,
    )

    original = get_model(0)
    persister.save_weights(original, step=0)

    remote_model_path = remote_root / "models" / "0" / "model.eqx"
    assert remote_model_path.exists()

    updated = get_model(1)
    loaded = persister.load_weights(updated, step=0)

    chex.assert_trees_all_equal(loaded, original)
    client.log_artifacts.assert_called_once()
    client.download_artifacts.assert_called_once()
    persister.cleanup()


def test_mlflow_persister_registers_versions(tmp_path: Path, mlflow_client_mock: tuple[MagicMock, Path]):
    """Model versions are registered when a name is provided."""
    client, _ = mlflow_client_mock
    client.search_registered_models.return_value = []

    persister = MLFlowPersister(
        client=client,
        run_id="run_123",
        artifact_path="models",
        model_framework=ModelFramework.Equinox,
        registered_model_name="TestModel",
    )

    persister.save_weights(get_model(2), step=5)

    client.create_registered_model.assert_called_once_with("TestModel")
    client.create_model_version.assert_called_once()
    call_kwargs = client.create_model_version.call_args.kwargs
    assert call_kwargs["name"] == "TestModel"
    assert call_kwargs["source"] == "runs:/run_123/models/5"
    assert call_kwargs["run_id"] == "run_123"
    persister.cleanup()


def test_mlflow_persister_from_logger_reuses_run(tmp_path: Path, mlflow_client_mock: tuple[MagicMock, Path]):
    """Persister created from logger uses existing client/run without terminating it."""

    class DummyLogger(MLFlowLogger):
        def __init__(self, client: MagicMock):
            self._client = client
            self._run_id = "run_123"

        @property
        def client(self) -> MagicMock:
            return self._client

        @property
        def run_id(self) -> str:
            return self._run_id

    client, remote_root = mlflow_client_mock
    logger = DummyLogger(client)

    persister = MLFlowPersister.from_logger(
        logger,
        artifact_path="models",
        model_framework=ModelFramework.Equinox,
    )

    original = get_model(0)
    persister.save_weights(original, step=1)

    remote_model_path = remote_root / "models" / "1" / "model.eqx"
    assert remote_model_path.exists()

    restored = persister.load_weights(get_model(1), step=1)
    chex.assert_trees_all_equal(restored, original)

    persister.cleanup()
    assert not client.set_terminated.called
