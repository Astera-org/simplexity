"""Tests for the torch_nn_utils module."""

import pytest
import torch

from simplexity.utils.torch_nn_utils import extract_learning_rates, snapshot_gradients, snapshot_named_parameters


@pytest.fixture
def model() -> torch.nn.Module:
    """Fixture for a test model."""
    return torch.nn.Linear(1, 1)


@pytest.fixture
def optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Fixture for a test optimizer."""
    return torch.optim.Adam(model.parameters(), lr=0.1)


def test_extract_learning_rates(optimizer: torch.optim.Optimizer):
    """Test extract_learning_rates function."""
    assert extract_learning_rates(optimizer) == {"group_0": 0.1}


def test_extract_learning_rates_multiple_param_groups():
    """Test extract_learning_rates function with multiple param groups."""
    multi_model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    multi_optimizer = torch.optim.Adam(
        [
            {"params": multi_model[0].parameters(), "lr": 0.01, "name": "first"},
            {"params": multi_model[1].parameters(), "lr": 0.02},
        ]
    )
    assert extract_learning_rates(multi_optimizer) == {"first": 0.01, "group_1": 0.02}


def test_snapshot_gradients(model: torch.nn.Module):
    """Test snapshot_gradients function."""
    assert not snapshot_gradients(model)

    expected = {}
    for name, param in model.named_parameters():
        grad = torch.randn_like(param.data)
        param.grad = grad
        expected[name] = grad.detach().clone()
    assert snapshot_gradients(model) == expected


def test_snapshot_named_parameters(model: torch.nn.Module):
    """Test snapshot_named_parameters function."""
    assert snapshot_named_parameters(model)
