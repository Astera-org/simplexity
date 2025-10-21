"""Utilities for converting between JAX arrays and PyTorch tensors on GPU.

This module provides functions to convert between JAX and PyTorch tensors
using DLPack, which allows for zero-copy GPU-to-GPU transfers without
going through CPU memory.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

try:
    import torch
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    """Convert a JAX array to PyTorch tensor using DLPack for GPU arrays.

    This function uses DLPack for zero-copy conversion when the JAX array
    is on GPU, avoiding expensive CPU transfers.

    Args:
        jax_array: JAX array to convert

    Returns:
        PyTorch tensor

    Raises:
        ImportError: If JAX or PyTorch is not available
    """
    try:
        torch_tensor = torch_dlpack.from_dlpack(jax_array)
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


def resolve_device(device_spec: str | None = "auto") -> str:
    """Resolve device specification to actual PyTorch device string.

    Args:
        device_spec: One of "auto", "cuda", "mps", "cpu", or None (treated as "auto")

    Returns:
        Resolved device string: "cuda", "mps", or "cpu"

    Raises:
        ValueError: If device_spec is not a recognized device type
        RuntimeError: If a specific device is requested but unavailable

    Examples:
        >>> resolve_device("auto")  # On CUDA machine
        'cuda'
        >>> resolve_device("cpu")
        'cpu'
    """
    if device_spec is None or device_spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    if device_spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")
        return "cuda"

    if device_spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but MPS is not available")
        return "mps"

    if device_spec == "cpu":
        return "cpu"

    raise ValueError(f"Unknown device specification: {device_spec}")
