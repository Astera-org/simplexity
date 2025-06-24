import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
from torch.nn import Transformer

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.configs.training.optimizer.config import PytorchAdamConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.evaluation.evaluate_pytorch_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister
from simplexity.training.train_pytorch_model import train


@pytest.fixture
def model() -> Transformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Transformer(
        d_model=64,
        nhead=1,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False,
        device=device,
    )


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
def test_train(model: Transformer, tmp_path: Path):
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
            name="pytorch_adam",
            instance=PytorchAdamConfig(
                _target_="torch.optim.AdamW",
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False,
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
    persister = LocalPytorchPersister(directory=str(tmp_path))

    original_metrics = evaluate(model=model, cfg=validation_cfg, data_generator=data_generator)
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
