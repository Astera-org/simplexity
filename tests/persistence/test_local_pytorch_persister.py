from pathlib import Path

import pytest
import torch
import torch.nn as nn

from simplexity.persistence.local_pytorch_persister import LocalPytorchPersister


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


def get_model(seed: int) -> SimpleLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    vocab_size = 2  # Binary vocab
    model = SimpleLM(vocab_size=vocab_size)
    return model.to(device)


def models_equal(model1: nn.Module, model2: nn.Module) -> bool:
    """Check if two PyTorch models have identical parameters."""
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    if set(params1.keys()) != set(params2.keys()):
        return False

    for name in params1.keys():
        if not torch.allclose(params1[name], params2[name]):
            return False

    return True


def test_local_persister(tmp_path: Path):
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
