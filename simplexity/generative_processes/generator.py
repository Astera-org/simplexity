from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess

State = TypeVar("State")


def batch_state(state: State, batch_size: int) -> State:
    """Batch a state using PyTree operations.

    Args:
        state: Initial state (can be array, tuple of arrays, etc.)
        batch_size: Number of batch repetitions

    Returns:
        Batched state with each leaf repeated batch_size times
    """
    return jax.tree_util.tree_map(lambda s: jnp.repeat(s[None, ...], batch_size, axis=0), state)


@eqx.filter_jit
def generate_data_batch(
    gen_states: State,
    data_generator: GenerativeProcess[State],
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[State, jax.Array, jax.Array]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    if bos_token is not None:
        tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), tokens], axis=1)
    if eos_token is not None:
        tokens = jnp.concatenate([tokens, jnp.full((batch_size, 1), eos_token)], axis=1)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    return gen_states, inputs, labels
