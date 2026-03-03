"""Test the local pytorch persister."""

from pathlib import Path

import torch
from torch.nn import GRU, Embedding, Linear, Module

from simplexity.tracking.model_persistence.local_pytorch_persister import (
    LocalPytorchPersister,
)


class SimpleLM(Module):
    """A simple language model for testing."""

    def __init__(self, vocab_size: int, embed_size: int = 64, hidden_size: int = 64):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.gru = GRU(embed_size, hidden_size, batch_first=True)
        self.output = Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        logits = self.output(gru_out)
        return logits


def get_model(seed: int) -> SimpleLM:
    """Get a model for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    vocab_size = 2  # Binary vocab
    model = SimpleLM(vocab_size=vocab_size)
    return model.to(device)


def models_equal(model1: Module, model2: Module) -> bool:
    """Check if two PyTorch models have identical parameters."""
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    if set(params1.keys()) != set(params2.keys()):
        return False

    return all(torch.allclose(params1[name], params2[name]) for name in params1)


def test_local_persister(tmp_path: Path):
    """Test the local pytorch persister."""
    directory = tmp_path
    filename = "test_model.pt"
    persister = LocalPytorchPersister(directory, filename)
    assert persister.directory == directory
    assert persister.filename == filename

    model = get_model(0)
    assert not (tmp_path / "0" / filename).exists()
    persister.save_weights(model, 0)
    assert (tmp_path / "0" / filename).exists()

    new_model = get_model(1)
    # Models with different seeds should have different parameters
    assert not models_equal(new_model, model)

    loaded_model = persister.load_weights(new_model, 0)
    # After loading, models should have the same parameters
    assert models_equal(loaded_model, model)
