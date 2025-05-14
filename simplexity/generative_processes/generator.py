import equinox as eqx
import jax

from simplexity.generative_processes.generative_process import GenerativeProcess


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, obs = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    inputs = obs[:, :-1]
    labels = obs[:, 1:]
    return gen_states, inputs, labels
