from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from simplexity.exceptions import DeviceResolutionError
from simplexity.utils.pytorch_utils import jax_to_torch, resolve_device, torch_to_jax


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_to_torch(device: str):
    """Test conversion of multidimensional JAX array."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    jax_device = jax.devices("gpu")[0] if device == "cuda" else jax.devices("cpu")[0]
    jax_array = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32, device=jax_device)
    torch_tensor = jax_to_torch(jax_array)

    assert isinstance(torch_tensor, torch.Tensor)
    assert torch_tensor.shape == (2, 2)
    assert torch_tensor.dtype == torch.float32
    np.testing.assert_array_equal(torch_tensor.cpu().numpy(), jax_array)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_to_jax(device: str):
    """Test conversion of multidimensional PyTorch tensor."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
    jax_array = torch_to_jax(torch_tensor)

    assert isinstance(jax_array, jax.Array)
    assert jax_array.shape == (2, 2)
    assert jax_array.dtype == jnp.float32
    np.testing.assert_array_equal(jax_array, torch_tensor.cpu().numpy())


def test_resolve_device_valid():
    """Test resolving a valid device specification."""
    with patch("torch.cuda.is_available") as mock_is_cuda_available:
        mock_is_cuda_available.return_value = True
        assert resolve_device("cuda") == "cuda"

    with patch("torch.backends.mps.is_available") as mock_is_mps_available:
        mock_is_mps_available.return_value = True
        assert resolve_device("mps") == "mps"

    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize("arg", ["auto", None])
def test_resolve_device_auto(arg: str | None):
    """Test resolving an auto device specification."""
    with patch("torch.cuda.is_available") as mock_is_cuda_available:
        mock_is_cuda_available.return_value = True
        assert resolve_device(arg) == "cuda"

    with (
        patch("torch.cuda.is_available") as mock_is_cuda_available,
        patch("torch.backends.mps.is_available") as mock_is_mps_available,
    ):
        mock_is_cuda_available.return_value = False
        mock_is_mps_available.return_value = True
        assert resolve_device(arg) == "mps"

    with (
        patch("torch.cuda.is_available") as mock_is_cuda_available,
        patch("torch.backends.mps.is_available") as mock_is_mps_available,
    ):
        mock_is_cuda_available.return_value = False
        mock_is_mps_available.return_value = False
        assert resolve_device(arg) == "cpu"


def test_resolve_device_unavailable():
    """Test resolving an unavailable device specification."""
    with patch("torch.cuda.is_available") as mock_is_cuda_available:
        mock_is_cuda_available.return_value = False
        with pytest.raises(DeviceResolutionError, match="CUDA requested but CUDA is not available"):
            resolve_device("cuda")

    with patch("torch.backends.mps.is_available") as mock_is_mps_available:
        mock_is_mps_available.return_value = False
        with pytest.raises(DeviceResolutionError, match="MPS requested but MPS is not available"):
            resolve_device("mps")


def test_resolve_device_unknown():
    """Test resolving an unknown device specification."""
    with pytest.raises(DeviceResolutionError, match="Unknown device specification: invalid"):
        resolve_device("invalid")
