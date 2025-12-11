"""Test the S3 tracker."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import chex
import equinox as eqx
import jax
import pytest

from simplexity.predictive_models.types import ModelFramework
from simplexity.tracking.model_persistence.local_equinox_persister import (
    LocalEquinoxPersister,
)
from simplexity.tracking.s3_tracker import S3Tracker
from tests.tracking.s3_mocks import MockBoto3Session, MockS3Client


@pytest.fixture(autouse=True)
def mock_boto():
    """Mock boto3 and botocore if missing."""
    if "boto3" not in sys.modules:
        sys.modules["boto3"] = MagicMock()
        sys.modules["boto3.session"] = MagicMock()
    if "botocore" not in sys.modules:
        sys.modules["botocore"] = MagicMock()
        sys.modules["botocore.exceptions"] = MagicMock()
        # Mock ClientError
        sys.modules["botocore.exceptions"].ClientError = Exception


def get_model(seed: int = 0) -> eqx.Module:
    """Get a model for testing."""
    return eqx.nn.Linear(in_features=4, out_features=2, key=jax.random.key(seed))


def test_s3_tracker(tmp_path: Path):
    """Test S3Tracker initialization."""
    s3_client_mock = MockS3Client(tmp_path)
    temp_dir = tempfile.TemporaryDirectory()
    with temp_dir:
        local_persister = LocalEquinoxPersister(temp_dir.name)
        tracker = S3Tracker(
            bucket="test_bucket",
            prefix="test_prefix",
            s3_client=s3_client_mock,
            temp_dir=temp_dir,
            local_persisters={ModelFramework.EQUINOX: local_persister},
        )
        assert tracker.bucket == "test_bucket"
        assert tracker.prefix == "test_prefix"

        model = get_model(0)
        assert not (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()
        tracker.save_model(model, 0)
        assert (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()

        new_model = get_model(1)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(new_model, model)
        loaded_model = tracker.load_model(new_model, 0)
        chex.assert_trees_all_equal(loaded_model, model)


def test_s3_tracker_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test S3Tracker.from_config with mocked Boto3 session."""

    def mock_session_init(profile_name=None, **kwargs):  # pylint: disable=unused-argument
        """Mock session initialization."""
        return MockBoto3Session.create(tmp_path)

    # Patch where it's imported (or sys.modules)
    monkeypatch.setattr("boto3.session.Session", mock_session_init)

    # Create config.ini file for testing
    config_file = tmp_path / "test_config.ini"
    config_content = """[aws]
profile_name = default

[s3]
bucket = test_bucket
"""
    config_file.write_text(config_content)

    tracker = S3Tracker.from_config(prefix="test_prefix", config_filename=str(config_file))

    assert tracker.bucket == "test_bucket"
    assert tracker.prefix == "test_prefix"

    model = get_model(0)
    assert not (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()
    tracker.save_model(model, 0)
    assert (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = tracker.load_model(new_model, 0)
    chex.assert_trees_all_equal(loaded_model, model)
