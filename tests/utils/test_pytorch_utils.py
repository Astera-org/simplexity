import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.utils.pytorch_utils import jax_to_torch, torch_to_jax

try:
    import torch
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


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
