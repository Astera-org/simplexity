import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
import torch.nn as nn

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.configs.training.optimizer.config import PytorchAdamConfig
from simplexity.evaluation.evaluate_pytorch_model import evaluate
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.logging.file_logger import FileLogger
from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister
from simplexity.training.train_pytorch_model import train


class DecoderOnlyTransformer(nn.Module):
    """A decoder-only transformer for language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.max_seq_len = 1024
        self.pos_embedding = nn.Embedding(self.max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(positions)
        embeddings = self.dropout(token_emb + pos_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        output = self.transformer_decoder(
            tgt=embeddings, memory=embeddings, tgt_mask=causal_mask, memory_mask=causal_mask
        )
        logits = self.output_projection(output)
        return logits


@pytest.fixture
def model() -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 2
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
    )
    return model.to(device)


def extract_losses(log_file_path: Path) -> jax.Array:
    """Extract losses from the log file."""
    with open(log_file_path) as f:
        log_lines = f.readlines()

    loss_pattern = r"'loss': ([\d.]+)"
    losses = []
    for line in log_lines:
        match = re.search(loss_pattern, line)
        if match:
            losses.append(float(match.group(1)))
    return jnp.array(losses)


@pytest.mark.slow
def test_train(model: torch.nn.Module, tmp_path: Path):
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

    original_metrics = evaluate(model=model, cfg=validation_cfg, data_generator=data_generator, bos_token=None, eos_token=None)
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
        training_bos_token=None,
        training_eos_token=None,
        validation_bos_token=None,
        validation_eos_token=None,
    )
    assert loss > 0.0
    losses = extract_losses(log_file_path)
    assert training_cfg.log_every is not None
    assert losses.shape == (training_cfg.num_steps // training_cfg.log_every,)
    final_metrics = evaluate(model=model, cfg=validation_cfg, data_generator=data_generator, bos_token=None, eos_token=None)
    assert final_metrics["loss"] < original_metrics["loss"]
    assert final_metrics["accuracy"] >= original_metrics["accuracy"]
    assert final_metrics["accuracy"] <= 1.0
