from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.predictive_models.gru_rnn import GRURNN


def get_model(seed: int) -> GRURNN:
    return GRURNN(in_size=1, out_size=2, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_local_persister(tmp_path: Path):
    directory = tmp_path
    persister = LocalEquinoxPersister(directory)
    assert persister.directory == directory

    model = get_model(0)
    assert not (tmp_path / "0" / "model.eqx").exists()
    persister.save_weights(model, 0)
    assert (tmp_path / "0" / "model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, 0)
    chex.assert_trees_all_equal(loaded_model, model)
