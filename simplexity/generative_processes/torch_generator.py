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
) -> tuple[jax.Array, torch.Tensor, torch.Tensor]:
    """Generate a batch of data."""
    gen_states, inputs, labels = generate_jax_data_batch(
        gen_states,
        data_generator,
        batch_size,
        sequence_len,
        key,
        bos_token,
        eos_token,
    )
    return gen_states, jax_to_torch(inputs), jax_to_torch(labels)
