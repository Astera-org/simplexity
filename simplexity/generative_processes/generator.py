import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, obs = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    if bos_token is not None:
        obs = jnp.concatenate([jnp.full((batch_size, 1), bos_token), obs], axis=1)
    if eos_token is not None:
        obs = jnp.concatenate([obs, jnp.full((batch_size, 1), eos_token)], axis=1)
    inputs = obs[:, :-1]
    labels = obs[:, 1:]
    return gen_states, inputs, labels
