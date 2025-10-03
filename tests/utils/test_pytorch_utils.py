import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.utils.pytorch_utils import jax_to_torch, resolve_device, torch_to_jax

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


class TestResolveDevice:
    """Test resolve_device function."""

    def test_auto_mode_returns_valid_device(self):
        """Test auto mode returns a valid PyTorch device string."""
        device = resolve_device("auto")
        assert device in ("cuda", "mps", "cpu")

    def test_none_treated_as_auto(self):
        """Test None is treated as auto mode."""
        device = resolve_device(None)
        assert device in ("cuda", "mps", "cpu")

    def test_cpu_always_available(self):
        """Test CPU is always available."""
        device = resolve_device("cpu")
        assert device == "cpu"

    def test_cuda_when_available(self):
        """Test CUDA request when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = resolve_device("cuda")
        assert device == "cuda"

    def test_cuda_unavailable_raises_runtime_error(self):
        """Test CUDA request raises RuntimeError when CUDA unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test unavailable case")
        with pytest.raises(RuntimeError, match="CUDA requested but CUDA is not available"):
            resolve_device("cuda")

    def test_mps_when_available(self):
        """Test MPS request when MPS is available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        device = resolve_device("mps")
        assert device == "mps"

    def test_mps_unavailable_raises_runtime_error(self):
        """Test MPS request raises RuntimeError when MPS unavailable."""
        if torch.backends.mps.is_available():
            pytest.skip("MPS is available, cannot test unavailable case")
        with pytest.raises(RuntimeError, match="MPS requested but MPS is not available"):
            resolve_device("mps")

    def test_invalid_spec_raises_value_error(self):
        """Test invalid device spec raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device specification"):
            resolve_device("invalid_device")
