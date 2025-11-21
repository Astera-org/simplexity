"""Generator for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import equinox as eqx
import jax
import jax.numpy as jnp
import torch

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.utils.pytorch_utils import jax_to_torch


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
    to_torch: bool = False,
) -> tuple[jax.Array, jax.Array | torch.Tensor, jax.Array | torch.Tensor]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    if bos_token is not None:
        tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), tokens], axis=1)
    if eos_token is not None:
        tokens = jnp.concatenate([tokens, jnp.full((batch_size, 1), eos_token)], axis=1)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    if to_torch:
        return gen_states, jax_to_torch(inputs), jax_to_torch(labels)
    return gen_states, inputs, labels
