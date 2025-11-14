"""Test the predictive models types module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import pytest

from simplexity.predictive_models.types import (
    ModelFramework,
    get_model_framework,
)

if TYPE_CHECKING:
    import torch


class _DummyEqxModule(eqx.Module):  # pylint: disable=too-few-public-methods
    weight: float = 1.0


def test_get_model_framework_equinox_model():
    """Test the get_model_framework function with an equinox model."""
    model = _DummyEqxModule()
    assert get_model_framework(model) is ModelFramework.EQUINOX


def test_get_model_framework_handles_equinox_and_pytorch_in_sequence():
    """Test the get_model_framework function with an equinox and pytorch model in sequence."""
    torch = pytest.importorskip("torch", reason="torch is optional for simplexity")

    class _DummyTorchModule(torch.nn.Module):  # pylint: disable=too-few-public-methods
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return x

    eqx_model = _DummyEqxModule()
    torch_model = _DummyTorchModule()

    assert get_model_framework(eqx_model) is ModelFramework.EQUINOX
    assert get_model_framework(torch_model) is ModelFramework.PYTORCH
    # Repeat to confirm that dispatch stays consistent when frameworks alternate.
    assert get_model_framework(eqx_model) is ModelFramework.EQUINOX


def test_get_model_framework_unsupported_type():
    """Test the get_model_framework function with an unsupported type."""
    with pytest.raises(ValueError, match="Unsupported model framework"):
        get_model_framework(object())
