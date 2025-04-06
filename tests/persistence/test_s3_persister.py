import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import chex
import jax
import pytest

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.s3_persister import S3Persister
from simplexity.predictive_models.gru_rnn import GRURNN


class LocalMockS3Paginator:
    """Local filesystem implementation of S3 paginator for testing."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def paginate(self, Bucket: str, Prefix: str) -> Iterable[Mapping[str, Any]]:
        """Paginate over the objects in an S3 bucket."""
        bucket_dir = self.root_dir / Bucket
        bucket_dir.mkdir(exist_ok=True)
        return [{"Key": obj.name} for obj in bucket_dir.glob(f"{Prefix}/*")]


class LocalMockS3Client:
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

    def get_paginator(self, operation_name: str) -> LocalMockS3Paginator:
        """Get a paginator for the given operation."""
        return LocalMockS3Paginator(self.root_dir)


def get_model(seed: int = 0) -> GRURNN:
    return GRURNN(in_size=1, out_size=2, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_s3_persister(tmp_path: Path):
    """Test S3Persister initialization."""
    s3_client_mock = LocalMockS3Client(tmp_path)
    temp_dir = tempfile.TemporaryDirectory()
    local_persister = LocalEquinoxPersister(temp_dir.name)
    persister = S3Persister(
        bucket="test-bucket",
        prefix="models",
        s3_client=s3_client_mock,
        temp_dir=temp_dir,
        local_persister=local_persister,
    )
    assert persister.bucket == "test-bucket"
    assert persister.prefix == "models"

    model = get_model(0)
    assert not (tmp_path / "test-bucket" / "models" / "0" / "model.eqx").exists()
    persister.save_weights(model, 0)
    assert (tmp_path / "test-bucket" / "models" / "0" / "model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, 0)
    chex.assert_trees_all_equal(loaded_model, model)
