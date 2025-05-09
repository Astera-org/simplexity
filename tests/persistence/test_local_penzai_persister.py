from pathlib import Path

import chex
import jax
import pytest
from penzai import pz
from penzai.core.variables import UnboundVariableError
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from tests.assertions import assert_trees_different


def test_local_penzai_persister(tmp_path: Path):
    config = LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=1,
        embedding_dim=32,
        projection_dim=32,
        mlp_hidden_dim=32,
        num_decoder_blocks=1,
        vocab_size=32,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )

    sequences = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, config.vocab_size)
    inputs = pz.nx.wrap(sequences, "batch", "seq")

    persister = LocalPenzaiPersister(tmp_path)

    key_0 = jax.random.PRNGKey(0)
    model_0 = build_llamalike_transformer(config, init_base_rng=key_0)
    assert isinstance(model_0, PredictiveModel)
    outputs_0 = model_0(inputs)

    # Saving creates a new file in the directory
    assert not (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()
    persister.save_weights(model_0, step=0)
    assert (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()

    # Attempting to save with overwrite=False fails if the file already exists
    # and the original file still exists
    with pytest.raises(FileExistsError):
        persister.save_weights(model_0, 0, overwrite_existing=False)
    assert (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()

    # Loading a checkpoint successfully replicates the original model
    unbound_model = build_llamalike_transformer(config)
    assert isinstance(unbound_model, PredictiveModel)
    with pytest.raises(UnboundVariableError):
        unbound_model(inputs)

    loaded_model = persister.load_weights(unbound_model, step=0)
    loaded_outputs = loaded_model(inputs)  # type: ignore
    chex.assert_trees_all_equal(loaded_outputs, outputs_0)

    # Saving a checkpoint when the file already exists will overwrite if
    # overwrite_existing=True
    key_1 = jax.random.PRNGKey(1)
    model_1 = build_llamalike_transformer(config, init_base_rng=key_1)
    assert isinstance(model_1, PredictiveModel)
    outputs_1 = model_1(inputs)
    assert_trees_different(model_1, model_0)  # pyright: ignore
    assert_trees_different(outputs_1, outputs_0)
    persister.save_weights(model_1, step=0, overwrite_existing=True)
    assert (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()

    # The saved checkpoint now replicates the new model
    unbound_model = build_llamalike_transformer(config)
    assert isinstance(unbound_model, PredictiveModel)
    with pytest.raises(UnboundVariableError):
        unbound_model(inputs)

    loaded_model = persister.load_weights(unbound_model, step=0)
    loaded_outputs = loaded_model(inputs)  # type: ignore
    chex.assert_trees_all_equal(loaded_outputs, outputs_1)
