import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer
from penzai.nn.layer import Layer as PenzaiModel

from simplexity.configs.evaluation.config import Config
from simplexity.evaluation.evaluate_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.equinox import vmap_model
from simplexity.utils.penzai import use_penzai_model


@pytest.fixture
def penzai_model() -> PenzaiModel:
    model_cfg = LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=1,
        embedding_dim=64,
        projection_dim=64,
        mlp_hidden_dim=64,
        num_decoder_blocks=2,
        vocab_size=2,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    model = build_llamalike_transformer(model_cfg, init_base_rng=jax.random.PRNGKey(0))
    return model


@pytest.fixture
def equinox_model() -> PredictiveModel:
    model = build_gru_rnn(vocab_size=2, embedding_size=16, num_layers=2, hidden_size=4, seed=0)
    return model


def extract_losses(log_file_path: Path) -> jax.Array:
    """Extract losses from the log file."""
    with open(log_file_path) as f:
        log_lines = f.readlines()

    loss_pattern = r"'loss': Array\(([\d.]+)"
    losses = []
    for line in log_lines:
        match = re.search(loss_pattern, line)
        if match:
            losses.append(float(match.group(1)))
    return jnp.array(losses)


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["penzai", "equinox"])
def test_evaluate_penzai_model(model_type: str, tmp_path: Path, request: pytest.FixtureRequest):
    if model_type == "penzai":
        model = request.getfixturevalue("penzai_model")
        evaluate_model = use_penzai_model(evaluate)
    else:
        model = request.getfixturevalue("equinox_model")
        evaluate_model = vmap_model(evaluate)
    cfg = Config(seed=0, sequence_len=4, batch_size=2, num_steps=3, log_every=5)
    data_generator = build_hidden_markov_model("even_ones", p=0.5)
    metrics = evaluate_model(model=model, cfg=cfg, data_generator=data_generator)
    assert metrics["loss"] > 0.0
    assert metrics["accuracy"] >= 0.0
    assert metrics["accuracy"] <= 1.0
