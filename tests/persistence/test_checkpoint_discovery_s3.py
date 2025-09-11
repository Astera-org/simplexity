import tempfile
from pathlib import Path

import jax

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.s3_persister import S3Persister
from simplexity.predictive_models.gru_rnn import GRURNN
from tests.persistence.s3_mocks import MockS3Client


def get_model(seed: int = 0) -> GRURNN:
    return GRURNN(vocab_size=2, embedding_size=4, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_s3_checkpoint_discovery(tmp_path: Path):
    s3_client = MockS3Client(tmp_path)
    temp_dir = tempfile.TemporaryDirectory()
    local = LocalEquinoxPersister(temp_dir.name)
    persister = S3Persister(
        bucket="b",
        prefix="p",
        s3_client=s3_client,
        temp_dir=temp_dir,
        local_persister=local,
    )

    # No checkpoints initially
    assert persister.list_checkpoints() == []
    assert persister.latest_checkpoint() is None

    model = get_model(0)
    persister.save_weights(model, 0)
    persister.save_weights(model, 7)

    assert persister.list_checkpoints() == [0, 7]
    assert persister.latest_checkpoint() == 7
    assert persister.checkpoint_exists(0)
    # URI should include filename when available
    assert persister.uri_for_step(7) == "s3://b/p/7/model.eqx"

