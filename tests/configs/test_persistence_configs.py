from pathlib import Path

import hydra

from simplexity.configs.persistence.config import LocalPersisterConfig, S3PersisterConfig
from simplexity.persistence.local_persister import LocalPersister
from simplexity.persistence.s3_persister import S3Persister


def test_local_persister_config():
    config = LocalPersisterConfig(
        _target_="simplexity.persistence.local_persister.LocalPersister",
        base_dir="test_base_dir",
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, LocalPersister)


def test_s3_persister_config(tmp_path: Path):
    filename = tmp_path / "config.ini"
    with open(filename, "w") as f:
        f.write(
            """
            [aws]
            profile_name = default

            [s3]
            bucket = test_bucket
            prefix = test_prefix
            """
        )
    config = S3PersisterConfig(
        _target_="simplexity.persistence.s3_persister.S3Persister.from_config",
        filename=str(filename),
    )
    persister = hydra.utils.instantiate(config)
    assert isinstance(persister, S3Persister)
