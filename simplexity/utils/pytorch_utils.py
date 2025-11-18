"""Utilities for converting between JAX arrays and PyTorch tensors on GPU.

This module provides functions to convert between JAX and PyTorch tensors
using DLPack, which allows for zero-copy GPU-to-GPU transfers without
going through CPU memory.
"""

import warnings
from collections.abc import Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

from simplexity.exceptions import DeviceResolutionError


def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    """Convert a JAX array to PyTorch tensor using DLPack for GPU arrays.

    This function uses DLPack for zero-copy conversion when the JAX array
    is on GPU, avoiding expensive CPU transfers.

    Args:
        jax_array: JAX array to convert

    Returns:
        PyTorch tensor
    """
    try:
        torch_tensor = torch_dlpack.from_dlpack(jax_array)
        return torch_tensor

    except TypeError as e:
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
    """
    try:
        dlpack_tensor = torch_dlpack.to_dlpack(torch_tensor)  # type: ignore
        jax_array = jax_dlpack.from_dlpack(dlpack_tensor)
        return jax_array

    except TypeError as e:
        warnings.warn(
            f"DLPack conversion failed ({e}), falling back to numpy. This may cause GPU-to-CPU transfer.",
            UserWarning,
            stacklevel=2,
        )
        numpy_array = torch_tensor.detach().cpu().numpy()
        jax_array = jnp.array(numpy_array)
        return jax_array


def resolve_device(device_spec: str | None = "auto") -> str:
    """Resolve device specification to actual PyTorch device string."""
    if device_spec is None or device_spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    if device_spec == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise DeviceResolutionError("CUDA requested but CUDA is not available")

    if device_spec == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise DeviceResolutionError("MPS requested but MPS is not available")

    if device_spec == "cpu":
        return "cpu"

    raise DeviceResolutionError(f"Unknown device specification: {device_spec}")


def tensor_collection_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    """Compute an L2 norm across an iterable of tensors without stacking.

    We detach and cast to float to avoid autograd interactions and dtype
    mismatches, skip empty tensors, and accumulate sums on CPU. This mirrors
    concatenating the tensors and calling ``torch.linalg.norm`` but without the
    intermediate allocation, which keeps metric computations lightweight even
    when parameters/gradients live on different devices or have differing
    shapes.
    """
    total = 0.0
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        total += float(torch.sum(torch.square(tensor.detach().float())))
    return total**0.5


def named_tensor_distance(current: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float:
    """Compute an L2 distance between two named tensor collections.

    Parameters are compared by name without stacking tensors into a single
    structure, which keeps the computation memory-efficient. Tensors are
    detached, moved to CPU, and cast to float so that distance metrics stay off
    the autograd graph and remain robust even if current and reference tensors
    live on different devices or use different dtypes. Using built-in norms
    would require materializing aligned tensors (or concatenations) per
    parameter pair on the same device/dtype, which is both more memory hungry
    and brittle when models evolve.
    """
    total = 0.0
    for name, tensor in current.items():
        ref_tensor = reference.get(name)
        if ref_tensor is None or tensor.numel() == 0:
            continue
        curr_cpu = tensor.detach().float()
        diff = curr_cpu - ref_tensor.float()
        total += float(torch.sum(torch.square(diff)))
    return total**0.5
