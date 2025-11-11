from pathlib import Path

import chex
import jax
import pytest
from penzai import pz
from penzai.core.variables import UnboundVariableError
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer
from penzai.nn.layer import Layer as PenzaiModel

from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister


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

    key = jax.random.PRNGKey(0)
    model = build_llamalike_transformer(config, init_base_rng=key)
    assert isinstance(model, PenzaiModel)
    outputs = model(inputs)

    assert not (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()
    persister.save_weights(model, step=0)
    assert (tmp_path / "0" / "_CHECKPOINT_METADATA").exists()

    unbound_model = build_llamalike_transformer(config)
    assert isinstance(unbound_model, PenzaiModel)
    with pytest.raises(UnboundVariableError):
        unbound_model(inputs)

    loaded_model = persister.load_weights(unbound_model, step=0)
    new_outputs = loaded_model(inputs)  # type: ignore
    chex.assert_trees_all_equal(new_outputs, outputs)
