from pathlib import Path

import hydra
import pytest

from simplexity.configs.persistence.config import (
    LocalEquinoxPersisterConfig,
    LocalPenzaiPersisterConfig,
    S3PersisterConfig,
)
from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.persistence.s3_persister import S3Persister
from tests.persistence.s3_mocks import MockBoto3Session


def test_local_equinox_persister_config(tmp_path: Path):
    config = LocalEquinoxPersisterConfig(
        _target_="simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
        directory=str(tmp_path),
        filename="test_filename",
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, LocalEquinoxPersister)


def test_local_penzai_persister_config(tmp_path: Path):
    config = LocalPenzaiPersisterConfig(
        _target_="simplexity.persistence.local_penzai_persister.LocalPenzaiPersister",
        directory=str(tmp_path),
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, LocalPenzaiPersister)


def test_s3_persister_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def mock_session_init(profile_name=None, **kwargs):
        return MockBoto3Session.create(tmp_path)

    monkeypatch.setattr("simplexity.persistence.s3_persister.boto3.session.Session", mock_session_init)

    config = S3PersisterConfig(
        _target_="simplexity.persistence.s3_persister.S3Persister.from_config",
        bucket="test_bucket",
        prefix="test_prefix",
        profile_name="test_profile",
        model_framework="equinox",
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, S3Persister)
