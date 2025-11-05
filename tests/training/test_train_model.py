import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer
from penzai.nn.layer import Layer as PenzaiModel

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.instance_config import InstanceConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.evaluation.evaluate_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.train_model import train
from simplexity.utils.equinox import vmap_model
from simplexity.utils.penzai import use_penzai_model


@dataclass
class AdamConfig(InstanceConfig):
    learning_rate: float
    b1: float
    b2: float
    eps: float
    eps_root: float
    nesterov: bool


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
def test_train(model_type: str, tmp_path: Path, request: pytest.FixtureRequest):
    if model_type == "penzai":
        penzai_model: PenzaiModel = request.getfixturevalue("penzai_model")
        model = penzai_model
        evaluate_model = use_penzai_model(evaluate)
    else:
        equinox_model: PredictiveModel = request.getfixturevalue("equinox_model")
        model = equinox_model
        evaluate_model = vmap_model(evaluate)
    data_generator = build_hidden_markov_model("zero_one_random", p=0.5)
    log_file_path = tmp_path / "test.log"
    logger = FileLogger(file_path=str(log_file_path))

    training_cfg = TrainConfig(
        seed=0,
        sequence_len=32,
        batch_size=64,
        num_steps=100,
        log_every=50,
        validate_every=75,
        checkpoint_every=100,
        optimizer=OptimizerConfig(
            name="optax_adam",
            instance=AdamConfig(
                _target_="optax.adam",
                learning_rate=0.001,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                eps_root=0.0,
                nesterov=True,
            ),
        ),
    )
    validation_cfg = ValidateConfig(
        seed=0,
        sequence_len=32,
        batch_size=64,
        num_steps=10,
        log_every=-1,
    )
    persister = LocalPenzaiPersister(directory=str(tmp_path))

    original_metrics = evaluate_model(model=model, cfg=validation_cfg, data_generator=data_generator)
    assert original_metrics["loss"] > 0.0
    assert original_metrics["accuracy"] >= 0.0
    assert original_metrics["accuracy"] <= 1.0
    model, loss = train(
        model,
        training_cfg,
        data_generator,
        logger,
        validation_cfg,
        data_generator,
        persister,
    )
    assert loss > 0.0
    losses = extract_losses(log_file_path)
    assert training_cfg.log_every is not None
    assert losses.shape == (training_cfg.num_steps // training_cfg.log_every,)
    final_metrics = evaluate_model(model=model, cfg=validation_cfg, data_generator=data_generator)
    assert final_metrics["loss"] < original_metrics["loss"]
    assert final_metrics["accuracy"] >= original_metrics["accuracy"]
    assert final_metrics["accuracy"] <= 1.0
