import equinox as eqx
import pytest

from simplexity.predictive_models.types import (
    ModelFramework,
    get_model_framework,
)


class _DummyEqxModule(eqx.Module):
    weight: float = 1.0


def test_get_model_framework_equinox_model():
    model = _DummyEqxModule()
    assert get_model_framework(model) is ModelFramework.Equinox


def test_get_model_framework_handles_equinox_and_pytorch_in_sequence():
    torch = pytest.importorskip("torch", reason="torch is optional for simplexity")

    class _DummyTorchModule(torch.nn.Module):
        def forward(self, x):
            return x

    eqx_model = _DummyEqxModule()
    torch_model = _DummyTorchModule()

    assert get_model_framework(eqx_model) is ModelFramework.Equinox
    assert get_model_framework(torch_model) is ModelFramework.Pytorch
    # Repeat to confirm that dispatch stays consistent when frameworks alternate.
    assert get_model_framework(eqx_model) is ModelFramework.Equinox


def test_get_model_framework_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported model framework"):
        get_model_framework(object())
