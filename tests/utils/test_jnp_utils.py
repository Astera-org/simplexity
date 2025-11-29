"""Tests for JAX NumPy utilities."""

from unittest.mock import create_autospec, patch

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.exceptions import DeviceResolutionError
from simplexity.utils.jnp_utils import log_matmul, resolve_jax_device, signed_logsumexp

_MOCK_CPU = create_autospec(jax.Device, instance=True)
_MOCK_GPU = create_autospec(jax.Device, instance=True)


def mock_devices(gpu_available: bool = True, cpu_available: bool = True):
    """Create a mock jax.devices function with configurable device availability.

    Args:
        gpu_available: Whether GPU devices should be available.
        cpu_available: Whether CPU devices should be available.

    Returns:
        A function that can be used as side_effect for patching jax.devices.
        The function returns the same mock device instances on each call, allowing
        for equality checks in tests.
    """

    cpu_devices = [_MOCK_CPU] if cpu_available else []
    gpu_devices = [_MOCK_GPU] if gpu_available else []

    def jax_devices(backend: str | None = None):
        """Mock jax.devices that returns mock CPU or GPU devices based on availability."""

        if backend == "cpu":
            return cpu_devices
        if backend == "gpu":
            return gpu_devices
        return gpu_devices + cpu_devices

    return jax_devices


def test_resolve_jax_device_specific_available():
    """Test resolve_jax_device function."""
    with (
        patch(
            "simplexity.utils.jnp_utils.jax.devices",
            side_effect=mock_devices(gpu_available=True, cpu_available=True),
        ),
    ):
        assert resolve_jax_device("cpu") == _MOCK_CPU
        assert resolve_jax_device("gpu") == _MOCK_GPU
        assert resolve_jax_device("cuda") == _MOCK_GPU


def test_resolve_jax_device_specific_unavailable():
    """Test resolve_jax_device function with unavailable specific devices."""
    with (
        patch(
            "simplexity.utils.jnp_utils.jax.devices",
            side_effect=mock_devices(gpu_available=False, cpu_available=False),
        ),
    ):
        with pytest.raises(DeviceResolutionError, match="CPU requested but not available"):
            resolve_jax_device("cpu")
        with pytest.raises(DeviceResolutionError, match="GPU requested but not available"):
            resolve_jax_device("gpu")
        with pytest.raises(DeviceResolutionError, match="GPU requested but not available"):
            resolve_jax_device("cuda")


@pytest.mark.parametrize("arg", ["auto", None])
@pytest.mark.parametrize(
    ("gpu_available", "cpu_available", "expected_device"),
    [
        (True, True, _MOCK_GPU),
        (True, False, _MOCK_GPU),
        (False, True, _MOCK_CPU),
    ],
)
def test_resolve_jax_device_auto(
    arg: str | None,
    gpu_available: bool,
    cpu_available: bool,
    expected_device: jax.Device,  # type: ignore[valid-type]
):
    """Test resolve_jax_device function with auto device specification."""
    with (
        patch(
            "simplexity.utils.jnp_utils.jax.devices",
            side_effect=mock_devices(gpu_available=gpu_available, cpu_available=cpu_available),
        ),
    ):
        assert resolve_jax_device(arg) == expected_device


@pytest.mark.parametrize("arg", ["auto", None])
def test_resolve_jax_device_auto_unavailable(arg: str | None):
    """Test resolve_jax_device function with auto device specification and unavailable devices."""
    with (
        patch(
            "simplexity.utils.jnp_utils.jax.devices",
            side_effect=mock_devices(gpu_available=False, cpu_available=False),
        ),
        pytest.raises(DeviceResolutionError, match="No devices available"),
    ):
        resolve_jax_device(arg)


def test_resolve_jax_device_invalid_backend():
    """Test resolve_jax_device function with invalid backend specification."""
    with (
        patch(
            "simplexity.utils.jnp_utils.jax.devices",
            side_effect=mock_devices(gpu_available=True, cpu_available=True),
        ),
        pytest.raises(DeviceResolutionError, match="Unknown device specification: invalid"),
    ):
        resolve_jax_device("invalid")


def test_log_matmul():
    """Test log_matmul function."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    mat_a = jax.random.uniform(key_a, (3, 4))
    mat_b = jax.random.uniform(key_b, (4, 5))
    actual = log_matmul(jnp.log(mat_a), jnp.log(mat_b))
    expected = jnp.log(mat_a @ mat_b)
    chex.assert_trees_all_close(actual, expected, atol=1e-7)


def test_signed_logsumexp():
    """Test signed_logsumexp function."""
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
