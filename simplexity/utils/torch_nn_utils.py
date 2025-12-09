"""Utilities for PyTorch neural network modules."""

import torch


def extract_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Extract learning rates from an optimizer."""
    rates: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        lr = float(group.get("lr", 0.0))
        rates[name] = lr
    return rates


def snapshot_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot gradients from a model."""
    gradients: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()
    return gradients


def snapshot_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot named parameters from a model."""
    return {name: param.detach().clone() for name, param in model.named_parameters()}
