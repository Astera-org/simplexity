from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.gru_rnn import GRURNN


def get_model(seed: int) -> GRURNN:
    return GRURNN(in_size=1, out_size=2, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_local_persister(tmp_path: Path):
    base_dir_str = str(tmp_path)
    persister = LocalPersister(base_dir_str)
    assert persister.base_dir == base_dir_str

    model = get_model(0)
    persister.save_weights(model, "test_model")
    assert (tmp_path / "test_model.eqx").exists()

    new_model = get_model(1)
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(new_model, model)
    loaded_model = persister.load_weights(new_model, "test_model")
    chex.assert_trees_all_equal(loaded_model, model)
