"""Torch generator for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import jax
import torch

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch as generate_jax_data_batch
from simplexity.utils.pytorch_utils import jax_to_torch


def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
    return_all_states: bool = False,
    return_prefix_probabilities: bool = False,
) -> tuple[jax.Array, jax.Array, torch.Tensor, torch.Tensor]:
    """Generate a batch of data with optional belief and probability outputs."""
    states, probs, inputs, labels = generate_jax_data_batch(
        gen_states,
        data_generator,
        batch_size,
        sequence_len,
        key,
        bos_token,
        eos_token,
        return_all_states,
        return_prefix_probabilities,
    )
    return states, probs, jax_to_torch(inputs), jax_to_torch(labels)
