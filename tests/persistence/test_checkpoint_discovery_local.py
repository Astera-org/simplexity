from pathlib import Path

import jax
import pytest

from simplexity.persistence.local_equinox_persister import LocalEquinoxPersister
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister
from simplexity.predictive_models.gru_rnn import GRURNN


def get_eqx_model(seed: int = 0) -> GRURNN:
    return GRURNN(vocab_size=2, embedding_size=4, hidden_sizes=[3, 3], key=jax.random.PRNGKey(seed))


def test_local_equinox_checkpoint_discovery(tmp_path: Path):
    persister = LocalEquinoxPersister(tmp_path, filename="model.eqx")
    model = get_eqx_model(0)

    # No checkpoints initially
    assert persister.list_checkpoints() == []
    assert persister.latest_checkpoint() is None
    assert not persister.checkpoint_exists(0)

    # Save a couple checkpoints
    persister.save_weights(model, 0)
    persister.save_weights(model, 5)

    assert persister.list_checkpoints() == [0, 5]
    assert persister.latest_checkpoint() == 5
    assert persister.checkpoint_exists(0)
    assert not persister.checkpoint_exists(1)
    assert persister.uri_for_step(5).startswith("file://")


def test_local_pytorch_checkpoint_discovery(tmp_path: Path):
    try:
        import torch  # type: ignore
    except Exception:
        pytest.skip("PyTorch not available")

    class Simple(torch.nn.Module):  # type: ignore
        def __init__(self):  # pragma: no cover - simple model definition
            super().__init__()
            self.l = torch.nn.Linear(2, 2)

        def forward(self, x):  # pragma: no cover - not used
            return self.l(x)

    persister = LocalPytorchPersister(tmp_path, filename="model.pt")
    model = Simple()

    assert persister.list_checkpoints() == []
    persister.save_weights(model, 0)
    persister.save_weights(model, 2)
    assert persister.list_checkpoints() == [0, 2]
    assert persister.latest_checkpoint() == 2
    assert persister.checkpoint_exists(2)
    assert persister.uri_for_step(2).startswith("file://")


def test_local_penzai_checkpoint_discovery(tmp_path: Path):
    try:
        from penzai.models.transformer.variants.llamalike_common import (
            LlamalikeTransformerConfig,
            build_llamalike_transformer,
        )
    except Exception:
        pytest.skip("Penzai not available")

    config = LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=1,
        embedding_dim=16,
        projection_dim=16,
        mlp_hidden_dim=16,
        num_decoder_blocks=1,
        vocab_size=8,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    model = build_llamalike_transformer(config, init_base_rng=jax.random.PRNGKey(0))

    persister = LocalPenzaiPersister(tmp_path)
    assert persister.list_checkpoints() == []
    persister.save_weights(model, 3)
    persister.save_weights(model, 7)
    assert persister.list_checkpoints() == [3, 7]
    assert persister.latest_checkpoint() == 7
    assert persister.checkpoint_exists(3)
    assert persister.uri_for_step(7).startswith("file://")
