import tempfile
from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.s3_persister import S3Persister
from simplexity.predictive_models.gru_rnn import GRURNN
from tests.persistence.s3_mocks import MockBoto3Session, MockS3Client


def get_model(seed: int = 0) -> GRURNN:
    return GRURNN(vocab_size=2, embedding_size=4, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_s3_persister(tmp_path: Path):
    """Test S3Persister initialization."""
    s3_client_mock = MockS3Client(tmp_path)
    temp_dir = tempfile.TemporaryDirectory()
    local_persister = LocalEquinoxPersister(temp_dir.name)
    persister = S3Persister(
        bucket="test_bucket",
        prefix="test_prefix",
        s3_client=s3_client_mock,
        temp_dir=temp_dir,
        local_persister=local_persister,
    )
    assert persister.bucket == "test_bucket"
    assert persister.prefix == "test_prefix"

    model = get_model(0)
    assert not (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()
    persister.save_weights(model, 0)
    assert (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, 0)
    chex.assert_trees_all_equal(loaded_model, model)


def test_s3_persister_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test S3Persister.from_config with mocked Boto3 session."""
    # Create a config file
    config_file = tmp_path / "config.ini"
    with open(config_file, "w") as f:
        f.write(
            """
            [aws]
            profile_name = default

            [s3]
            bucket = test_bucket
            """
        )

    def mock_session_init(profile_name=None, **kwargs):
        return MockBoto3Session.create(tmp_path)

    monkeypatch.setattr("simplexity.persistence.s3_persister.Session", mock_session_init)

    persister = S3Persister.from_config(str(config_file), "test_prefix")

    assert persister.bucket == "test_bucket"
    assert persister.prefix == "test_prefix"

    model = get_model(0)
    assert not (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()
    persister.save_weights(model, 0)
    assert (tmp_path / "test_bucket" / "test_prefix" / "0" / "model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, 0)
    chex.assert_trees_all_equal(loaded_model, model)
