import shutil
from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.s3_persister import S3ModelPersister
from simplexity.predictive_models.gru_rnn import GRURNN


class LocalS3MockClient:
    """Local filesystem implementation of S3 client for testing."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def upload_file(self, file_name: str, bucket: str, object_name: str) -> None:
        """Copy file to mock S3 storage."""
        bucket_dir = self.root_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        target_path = bucket_dir / object_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, target_path)

    def download_file(self, bucket: str, object_name: str, file_name: str) -> None:
        """Copy file from mock S3 storage."""
        source_path = self.root_dir / bucket / object_name
        if not source_path.exists():
            raise RuntimeError(f"File not found: {object_name}")
        shutil.copy(source_path, file_name)


def get_model(seed: int = 0) -> GRURNN:
    return GRURNN(in_size=1, out_size=2, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_s3_persister(tmp_path: Path):
    """Test S3ModelPersister initialization."""
    s3_client_mock = LocalS3MockClient(tmp_path)
    persister = S3ModelPersister(bucket="test-bucket", prefix="models", s3_client=s3_client_mock)
    assert persister.bucket == "test-bucket"
    assert persister.prefix == "models"

    model = get_model(0)
    persister.save_weights(model, "test_model")
    assert (tmp_path / "test-bucket" / "models" / "test_model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, "test_model")
    chex.assert_trees_all_equal(loaded_model, model)
