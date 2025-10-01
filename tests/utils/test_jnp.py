import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.utils.jnp import log_matmul, resolve_jax_device, signed_logsumexp


def test_log_matmul():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    A = jax.random.uniform(key_a, (3, 4))
    B = jax.random.uniform(key_b, (4, 5))
    actual = log_matmul(jnp.log(A), jnp.log(B))
    expected = jnp.log(A @ B)
    chex.assert_trees_all_close(actual, expected, atol=1e-7)


def test_signed_logsumexp():
    # values:
    # [ 4,  1,  2] ->  7
    # [-1, -3, -2] -> -6
    # [-1,  0,  2] ->  1

    log_abs_values = jnp.log(
        jnp.array(
            [
                [4, 1, 2],
                [1, 3, 2],
                [1, 0, 2],
            ]
        )
    )
    signs = jnp.array(
        [
            [1, 1, 1],
            [-1, -1, -1],
            [-1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    actual_log_abs_values, actual_signs = signed_logsumexp(log_abs_values, signs, axis=1)
    expected_log_abs_values = jnp.log(jnp.array([7, 6, 1]))
    expected_signs = jnp.array([1, -1, 1], dtype=jnp.int32)

    chex.assert_trees_all_close(actual_log_abs_values, expected_log_abs_values)
    chex.assert_trees_all_close(actual_signs, expected_signs)


class TestResolveJaxDevice:
    """Test resolve_jax_device function."""

    def test_auto_mode_returns_device(self):
        """Test auto mode returns a valid JAX device."""
        device = resolve_jax_device("auto")
        assert isinstance(device, jax.Device)

    def test_none_treated_as_auto(self):
        """Test None is treated as auto mode."""
        device = resolve_jax_device(None)
        assert isinstance(device, jax.Device)

    def test_cpu_returns_cpu_device(self):
        """Test explicit CPU request returns CPU device."""
        device = resolve_jax_device("cpu")
        assert isinstance(device, jax.Device)
        assert "cpu" in str(device).lower()

    def test_gpu_when_available(self):
        """Test GPU request when GPU is available."""
        try:
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                pytest.skip("GPU not available")
        except RuntimeError:
            pytest.skip("GPU not available")

        device = resolve_jax_device("gpu")
        assert isinstance(device, jax.Device)
        assert "gpu" in str(device).lower() or "cuda" in str(device).lower()

    def test_cuda_when_available(self):
        """Test CUDA request when GPU is available."""
        try:
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                pytest.skip("GPU not available")
        except RuntimeError:
            pytest.skip("GPU not available")

        device = resolve_jax_device("cuda")
        assert isinstance(device, jax.Device)
        assert "gpu" in str(device).lower() or "cuda" in str(device).lower()

    def test_gpu_unavailable_raises_runtime_error(self):
        """Test GPU request raises RuntimeError when GPU unavailable."""
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                pytest.skip("GPU is available, cannot test unavailable case")
        except RuntimeError:
            pass

        with pytest.raises(RuntimeError, match="GPU requested but no GPU devices available"):
            resolve_jax_device("gpu")

    def test_cuda_unavailable_raises_runtime_error(self):
        """Test CUDA request raises RuntimeError when GPU unavailable."""
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                pytest.skip("GPU is available, cannot test unavailable case")
        except RuntimeError:
            pass

        with pytest.raises(RuntimeError, match="GPU requested but no GPU devices available"):
            resolve_jax_device("cuda")

    def test_invalid_spec_raises_value_error(self):
        """Test invalid device spec raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device specification"):
            resolve_jax_device("invalid_device")

    def test_unknown_specs_raise_value_error(self):
        """Test various unknown specs raise ValueError."""
        invalid_specs = ["tpu", "gpu0", "cuda:0", "mps", "unknown"]
        for spec in invalid_specs:
            with pytest.raises(ValueError, match="Unknown device specification"):
                resolve_jax_device(spec)
