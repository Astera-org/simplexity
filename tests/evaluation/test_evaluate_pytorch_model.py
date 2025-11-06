import re  # noqa: I001
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
import torch.nn as nn

from omegaconf import DictConfig
from simplexity.evaluation.evaluate_pytorch_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model


class SimpleLM(nn.Module):
    """A simple language model for testing."""

    def __init__(self, vocab_size: int, embed_size: int = 64, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        logits = self.output(gru_out)
        return logits


@pytest.fixture
def model() -> SimpleLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 2  # Binary vocab for even_ones
    model = SimpleLM(vocab_size=vocab_size)
    return model.to(device)


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
def test_evaluate_pytorch_model(model: SimpleLM, tmp_path: Path):
    cfg = DictConfig({"seed": 0, "sequence_len": 4, "batch_size": 2, "num_steps": 3, "log_every": 5})
    data_generator = build_hidden_markov_model("even_ones", p=0.5)
    metrics = evaluate(model=model, cfg=cfg, data_generator=data_generator)
    assert metrics["loss"] > 0.0
    assert metrics["accuracy"] >= 0.0
    assert metrics["accuracy"] <= 1.0
