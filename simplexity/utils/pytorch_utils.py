"""Utilities for converting between JAX arrays and PyTorch tensors on GPU.

This module provides functions to convert between JAX and PyTorch tensors
using DLPack, which allows for zero-copy GPU-to-GPU transfers without
going through CPU memory.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack


def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    """Convert a JAX array to PyTorch tensor using DLPack for GPU arrays.

    This function uses DLPack for zero-copy conversion when the JAX array
    is on GPU, avoiding expensive CPU transfers.

    Args:
        jax_array: JAX array to convert
        device: Target PyTorch device (optional, will use JAX array's device if None)

    Returns:
        PyTorch tensor

    Raises:
        ImportError: If JAX or PyTorch is not available
    """
    try:
        dlpack_tensor = jax_dlpack.to_dlpack(jax_array)
        torch_tensor = torch_dlpack.from_dlpack(dlpack_tensor)
        return torch_tensor

    except Exception as e:
        warnings.warn(
            f"DLPack conversion failed ({e}), falling back to numpy. This may cause GPU-to-CPU transfer.",
            UserWarning,
            stacklevel=2,
        )
        numpy_array = np.array(jax_array)
        torch_tensor = torch.from_numpy(numpy_array)
        return torch_tensor


def torch_to_jax(torch_tensor: torch.Tensor) -> jax.Array:
    """Convert a PyTorch tensor to JAX array using DLPack for GPU tensors.

    This function uses DLPack for zero-copy conversion when the PyTorch tensor
    is on GPU, avoiding expensive CPU transfers.

    Args:
        torch_tensor: PyTorch tensor to convert

    Returns:
        JAX array

    Raises:
        ImportError: If JAX or PyTorch is not available
    """
    try:
        dlpack_tensor = torch_dlpack.to_dlpack(torch_tensor)  # type: ignore
        jax_array = jax_dlpack.from_dlpack(dlpack_tensor)
        return jax_array

    except Exception as e:
        warnings.warn(
            f"DLPack conversion failed ({e}), falling back to numpy. This may cause GPU-to-CPU transfer.",
            UserWarning,
            stacklevel=2,
        )
        numpy_array = torch_tensor.detach().cpu().numpy()
        jax_array = jnp.array(numpy_array)
        return jax_array
