import re
from pathlib import Path

import jax
import jax.numpy as jnp

from simplexity.configs.train.config import Config as TrainConfig
from simplexity.configs.train.optimizer.config import AdamConfig
from simplexity.configs.train.optimizer.config import Config as OptimizerConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_persister import LocalPersister
from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.training.train import train


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


def test_train(tmp_path: Path):
    generative_process = build_hidden_markov_model("even_ones", p=0.5)
    model = build_gru_rnn(generative_process.vocab_size, num_layers=2, hidden_size=4, seed=0)
    persister = LocalPersister(base_dir=str(tmp_path))
    log_file_path = tmp_path / "test.log"
    logger = FileLogger(file_path=str(log_file_path))

    cfg = TrainConfig(
        seed=0,
        sequence_len=4,
        batch_size=2,
        num_steps=8,
        log_every=1,
        validate_every=1,
        num_validation_steps=1,
        checkpoint_every=8,
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
    model = train(cfg, model, generative_process, generative_process, persister, logger)
    losses = extract_losses(log_file_path)
    assert losses.shape == (cfg.num_steps,)
