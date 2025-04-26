import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.configs.training.optimizer.config import AdamConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.evaluation.evaluate_penzai_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_penzai_persister import LocalPenzaiPersister
from simplexity.training.train_penzai_model import train


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
def test_train(tmp_path: Path):
    data_generator = build_hidden_markov_model("zero_one_random", p=0.5)
    model_cfg = LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=1,
        embedding_dim=64,
        projection_dim=64,
        mlp_hidden_dim=64,
        num_decoder_blocks=2,
        vocab_size=data_generator.vocab_size,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    model = build_llamalike_transformer(model_cfg, init_base_rng=jax.random.PRNGKey(0))
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
        checkpoint_name="test",
        optimizer=OptimizerConfig(
            name="adam",
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

    original_metrics = evaluate(model, validation_cfg, data_generator)
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
    assert losses.shape == (training_cfg.num_steps // training_cfg.log_every,)
    final_metrics = evaluate(model, validation_cfg, data_generator)
    assert final_metrics["loss"] < original_metrics["loss"]
    assert final_metrics["accuracy"] >= original_metrics["accuracy"]
    assert final_metrics["accuracy"] <= 1.0
