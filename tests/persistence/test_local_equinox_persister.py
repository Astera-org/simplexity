from pathlib import Path

import chex
import jax
import pytest

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.predictive_models.gru_rnn import GRURNN
from tests.assertions import assert_trees_different


def get_model(seed: int) -> GRURNN:
    return GRURNN(in_size=1, out_size=2, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_local_persister(tmp_path: Path):
    directory = tmp_path
    filename = "test_model.eqx"
    persister = LocalEquinoxPersister(directory, filename)
    assert persister.directory == directory
    assert persister.filename == filename

    # Saving creates a new file in the directory
    model_0 = get_model(0)
    assert not (tmp_path / "0" / filename).exists()
    persister.save_weights(model_0, 0)
    assert (tmp_path / "0" / filename).exists()

    # Attempting to save with overwrite=False fails if the file already exists
    # and the original file still exists
    with pytest.raises(FileExistsError):
        persister.save_weights(model_0, 0, overwrite_existing=False)
    assert (tmp_path / "0" / filename).exists()

    # Loading a checkpoint successfully replicates the original model
    model_1 = get_model(1)
    assert_trees_different(model_1, model_0)  # pyright: ignore
    loaded_model = persister.load_weights(model_1, 0)
    chex.assert_trees_all_equal(loaded_model, model_0)

    # Saving a checkpoint when the file already exists will overwrite if
    # overwrite_existing=True
    model_1 = get_model(1)
    assert_trees_different(model_1, model_0)  # pyright: ignore
    assert (tmp_path / "0" / filename).exists()
    persister.save_weights(model_1, 0, overwrite_existing=True)
    assert (tmp_path / "0" / filename).exists()

    # The saved checkpoint now replicates the new model
    model_2 = get_model(2)
    assert_trees_different(model_2, model_1)  # pyright: ignore
    loaded_model = persister.load_weights(model_2, 0)
    chex.assert_trees_all_equal(loaded_model, model_1)
