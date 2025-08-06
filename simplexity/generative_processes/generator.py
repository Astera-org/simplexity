import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProtocol


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProtocol,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
