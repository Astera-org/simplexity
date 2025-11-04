from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.persistence.s3_persister import S3Persister
from tests.persistence.s3_mocks import MockBoto3Session


def test_local_equinox_persister_config(tmp_path: Path):
    config = DictConfig(
        {
            "_target_": "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
            "directory": str(tmp_path),
            "filename": "test_filename",
        }
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, LocalEquinoxPersister)


def test_local_penzai_persister_config(tmp_path: Path):
    config = DictConfig(
        {
            "_target_": "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister",
            "directory": str(tmp_path),
        }
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, LocalPenzaiPersister)


def test_s3_persister_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def mock_session_init(profile_name=None, **kwargs):
        return MockBoto3Session.create(tmp_path)

    monkeypatch.setattr("simplexity.persistence.s3_persister.boto3.session.Session", mock_session_init)

    # Create config.ini file for testing
    config_file = tmp_path / "test_config.ini"
    config_content = """[aws]
profile_name = test_profile

[s3]
bucket = test_bucket
"""
    config_file.write_text(config_content)

    config = DictConfig(
        {
            "_target_": "simplexity.persistence.s3_persister.S3Persister.from_config",
            "prefix": "test_prefix",
            "model_framework": "equinox",
            "config_filename": str(config_file),
        }
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, S3Persister)
