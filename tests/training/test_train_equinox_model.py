import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from simplexity.configs.evaluation.config import Config as ValidationConfig
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.training.optimizer.config import AdamConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.evaluation.evaluate_equinox_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.training.train_equinox_model import train


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
    data_generator = build_hidden_markov_model("even_ones", p=0.5)
    model = build_gru_rnn(data_generator.vocab_size, num_layers=2, hidden_size=4, seed=0)
    log_file_path = tmp_path / "test.log"
    logger = FileLogger(file_path=str(log_file_path))
    training_cfg = TrainingConfig(
        seed=0,
        sequence_len=4,
        batch_size=128,
        num_steps=100,
        log_every=10,
        validate_every=80,
        checkpoint_every=80,
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
    validation_cfg = ValidationConfig(
        seed=0,
        sequence_len=4,
        batch_size=128,
        num_steps=8,
        log_every=1,
    )
    persister = LocalPersister(base_dir=str(tmp_path))
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
