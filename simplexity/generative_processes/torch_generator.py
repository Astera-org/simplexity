"""Torch generator for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from typing import TypedDict

import jax
import torch

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import (
    DataBatch,
    generate_data_batch as generate_jax_data_batch,
    generate_data_batch_with_full_history as generate_jax_data_batch_with_full_history,
)
from simplexity.utils.pytorch_utils import jax_to_torch


class TorchDataBatch(TypedDict):
    """Torch payload with tensor tokens and JAX states."""

    gen_states: jax.Array | tuple[jax.Array, ...]
    inputs: torch.Tensor
    labels: torch.Tensor
    belief_states: jax.Array | tuple[jax.Array, ...]
    prefix_probabilities: jax.Array


def generate_data_batch(
    gen_states: jax.Array | tuple[jax.Array, ...],
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
    device: str | torch.device | None = None,
) -> TorchDataBatch:
    """Generate a batch of data.

    Args:
        gen_states: Generator states
        data_generator: Generative process
        batch_size: Batch size
        sequence_len: Sequence length
        key: JAX random key
        bos_token: Optional beginning of sequence token
        eos_token: Optional end of sequence token
        device: Optional target device for PyTorch tensors

    Returns:
        Dict containing generator states, belief/prefix fields, and torch inputs/labels
    """
    result = generate_jax_data_batch(
        gen_states,
        data_generator,
        batch_size,
        sequence_len,
        key,
        bos_token,
        eos_token,
    )
    inputs = result["inputs"]
    labels = result["labels"]
    assert isinstance(inputs, jax.Array)
    assert isinstance(labels, jax.Array)
    return TorchDataBatch(
        gen_states=result["gen_states"],
        belief_states=result["belief_states"],
        prefix_probabilities=result["prefix_probabilities"],
        inputs=jax_to_torch(inputs, device),
        labels=jax_to_torch(labels, device),
    )


def generate_data_batch_with_full_history(
    gen_states: jax.Array | tuple[jax.Array, ...],
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
    device: str | torch.device | None = None,
) -> TorchDataBatch:
    """Generate data plus full belief/prefix histories.

    Args:
        gen_states: Generator states
        data_generator: Generative process
        batch_size: Batch size
        sequence_len: Sequence length
        key: JAX random key
        bos_token: Optional beginning of sequence token
        eos_token: Optional end of sequence token
        device: Optional target device for PyTorch tensors

    Returns:
        TorchDataBatch with keys:
            - gen_states: Final generator state (jax.Array or tuple[jax.Array, ...])
            - belief_states: Belief states (jax.Array or tuple[jax.Array, ...])
            - prefix_probabilities: Prefix probabilities (jax.Array)
            - inputs: Input tokens (torch.Tensor)
            - labels: Label tokens (torch.Tensor)
    """
    result: DataBatch = generate_jax_data_batch_with_full_history(
        gen_states,
        data_generator,
        batch_size,
        sequence_len,
        key,
        bos_token,
        eos_token,
    )
    # Extract inputs and labels (these are always jax.Arrays)
    inputs = result["inputs"]
    labels = result["labels"]
    assert isinstance(inputs, jax.Array)
    assert isinstance(labels, jax.Array)

    return TorchDataBatch(
        gen_states=result["gen_states"],
        belief_states=result["belief_states"],
        prefix_probabilities=result["prefix_probabilities"],
        inputs=jax_to_torch(inputs, device),
        labels=jax_to_torch(labels, device),
    )
