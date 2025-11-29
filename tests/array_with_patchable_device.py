"""Utilities for testing with patchable device attributes on JAX arrays."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp


class ArrayWithPatchableDevice:
    """Wrapper that makes device attribute mutable for testing.

    JAX arrays have a read-only device property, which makes it difficult to test
    device mismatch scenarios. This wrapper provides a mutable device attribute
    while delegating all other operations to the underlying JAX array.
    """

    def __init__(self, array: jax.Array, device: Any):
        """Initialize wrapper with array and device.

        Args:
            array: The underlying JAX array
            device: The device to report (can be a mock for testing)
        """
        self._array = array
        self.device = device

    @property
    def array(self) -> jax.Array:
        """Get the underlying JAX array."""
        return self._array

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying array."""
        return getattr(self._array, name)

    def __array__(self, dtype: Any = None) -> Any:
        """Allow conversion to numpy array."""
        return self._array.__array__(dtype)

    def __ge__(self, other: Any) -> Any:
        """Delegate >= to underlying array."""
        return self._array >= other

    def __le__(self, other: Any) -> Any:
        """Delegate <= to underlying array."""
        return self._array <= other

    def __gt__(self, other: Any) -> Any:
        """Delegate > to underlying array."""
        return self._array > other

    def __lt__(self, other: Any) -> Any:
        """Delegate < to underlying array."""
        return self._array < other

    def __add__(self, other: Any) -> Any:
        """Delegate + to underlying array, preserving wrapper."""
        result = self._array + (other.array if isinstance(other, ArrayWithPatchableDevice) else other)
        return ArrayWithPatchableDevice(result, self.device)

    def __sub__(self, other: Any) -> Any:
        """Delegate - to underlying array, preserving wrapper."""
        result = self._array - (other.array if isinstance(other, ArrayWithPatchableDevice) else other)
        return ArrayWithPatchableDevice(result, self.device)

    def __mul__(self, other: Any) -> Any:
        """Delegate * to underlying array, preserving wrapper."""
        result = self._array * (other.array if isinstance(other, ArrayWithPatchableDevice) else other)
        return ArrayWithPatchableDevice(result, self.device)

    def __truediv__(self, other: Any) -> Any:
        """Delegate / to underlying array, preserving wrapper."""
        result = self._array / (other.array if isinstance(other, ArrayWithPatchableDevice) else other)
        return ArrayWithPatchableDevice(result, self.device)

    def __matmul__(self, other: Any) -> Any:
        """Delegate @ to underlying array, preserving wrapper."""
        result = self._array @ (other.array if isinstance(other, ArrayWithPatchableDevice) else other)
        return ArrayWithPatchableDevice(result, self.device)

    def __getitem__(self, key: Any) -> Any:
        """Delegate indexing to underlying array, preserving wrapper."""
        result = self._array[key]
        return ArrayWithPatchableDevice(result, self.device)


def extract_array(a: Any) -> Any:
    """Extract underlying array from wrapper if needed."""
    return a.array if isinstance(a, ArrayWithPatchableDevice) else a


@contextmanager
def patch_jax_for_patchable_device(module_path: str, mock_devices: tuple[Any, ...] = ()):
    """Context manager that patches JAX functions to work with ArrayWithPatchableDevice.

    This function patches all JAX functions that need to handle ArrayWithPatchableDevice
    wrappers, extracting the underlying array before calling the original function.

    Args:
        module_path: The module path where JAX functions are imported (e.g.,
            "simplexity.generative_processes.hidden_markov_model")
        mock_devices: Tuple of mock devices that should be handled specially in
            jnp.ones and jnp.zeros

    Yields:
        A context manager that applies all necessary patches
    """
    # Get original functions before patching to avoid recursion
    import jax.nn

    original_jnp_sum = jnp.sum
    original_jnp_all = jnp.all
    original_jnp_log = jnp.log
    original_jnp_max = jnp.max
    original_jnp_isclose = jnp.isclose
    original_jnp_linalg_eig = jnp.linalg.eig
    original_jnp_ones = jnp.ones
    original_jnp_zeros = jnp.zeros
    original_jnp_asarray = jnp.asarray
    original_jax_nn_logsumexp = jax.nn.logsumexp

    def sum_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jnp.sum."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jnp_sum(extract_array(a), **kwargs)
        return ArrayWithPatchableDevice(result, device) if was_wrapper else result

    def all_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jnp.all."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jnp_all(extract_array(a), **kwargs)
        return ArrayWithPatchableDevice(result, device) if was_wrapper else result

    def log_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jnp.log."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jnp_log(extract_array(a), **kwargs)
        return ArrayWithPatchableDevice(result, device) if was_wrapper else result

    def max_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jnp.max."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jnp_max(extract_array(a), **kwargs)
        return ArrayWithPatchableDevice(result, device) if was_wrapper else result

    def isclose_side_effect(a: Any, b: Any, **kwargs: Any) -> Any:
        """Extract underlying arrays from wrappers before calling original jnp.isclose."""
        # Comparison operations return booleans, not arrays, so no need to wrap
        return original_jnp_isclose(extract_array(a), extract_array(b), **kwargs)

    def eig_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jnp.linalg.eig."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jnp_linalg_eig(extract_array(a), **kwargs)
        # eig returns a tuple of (eigenvalues, eigenvectors)
        if was_wrapper and isinstance(result, tuple):
            eigenvalues, eigenvectors = result
            return (
                ArrayWithPatchableDevice(eigenvalues, device),
                ArrayWithPatchableDevice(eigenvectors, device),
            )
        return result

    def logsumexp_side_effect(a: Any, **kwargs: Any) -> Any:
        """Extract underlying array from wrapper before calling original jax.nn.logsumexp."""
        was_wrapper = isinstance(a, ArrayWithPatchableDevice)
        device = a.device if was_wrapper else None
        result = original_jax_nn_logsumexp(extract_array(a), **kwargs)
        return ArrayWithPatchableDevice(result, device) if was_wrapper else result

    def asarray_side_effect(a: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Convert ArrayWithPatchableDevice wrapper to underlying array for JAX operations."""
        if isinstance(a, ArrayWithPatchableDevice):
            return original_jnp_asarray(a.array, dtype=dtype, **kwargs)
        return original_jnp_asarray(a, dtype=dtype, **kwargs)

    def ones_side_effect(shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Any:
        """Handle mock devices by calling original jnp.ones without device parameter."""
        if device in mock_devices:
            # If device is a mock, create on default device
            result = original_jnp_ones(shape, dtype=dtype, **kwargs)
            # Return a wrapper with the mock device for consistency
            return ArrayWithPatchableDevice(result, device)
        return original_jnp_ones(shape, dtype=dtype, device=device, **kwargs)

    def zeros_side_effect(shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Any:
        """Handle mock devices by calling original jnp.zeros without device parameter."""
        if device in mock_devices:
            # If device is a mock, create on default device
            result = original_jnp_zeros(shape, dtype=dtype, **kwargs)
            # Return a wrapper with the mock device for consistency
            return ArrayWithPatchableDevice(result, device)
        return original_jnp_zeros(shape, dtype=dtype, device=device, **kwargs)

    with (
        patch("jax.numpy.asarray", side_effect=asarray_side_effect),
        patch("jax.nn.logsumexp", side_effect=logsumexp_side_effect),
        patch(f"{module_path}.jnp.sum", side_effect=sum_side_effect),
        patch(f"{module_path}.jnp.all", side_effect=all_side_effect),
        patch(f"{module_path}.jnp.log", side_effect=log_side_effect),
        patch(f"{module_path}.jnp.max", side_effect=max_side_effect),
        patch(f"{module_path}.jnp.isclose", side_effect=isclose_side_effect),
        patch(f"{module_path}.jnp.linalg.eig", side_effect=eig_side_effect),
        patch(f"{module_path}.jnp.ones", side_effect=ones_side_effect),
        patch(f"{module_path}.jnp.zeros", side_effect=zeros_side_effect),
    ):
        yield
