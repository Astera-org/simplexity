from collections import defaultdict

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from penzai import pz
from penzai.core.named_axes import NamedArray
from penzai.nn.layer import Layer

from simplexity.configs.validation.config import Config as ValidateConfig
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
) -> tuple[jax.Array, NamedArray, jax.Array]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, obs = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    named_inputs = pz.nx.wrap(obs[:, :-1], "batch", "seq")
    labels = obs[:, 1:]
    return gen_states, named_inputs, labels


@eqx.filter_jit
def loss_fn(model: Layer, named_inputs: NamedArray, labels: jax.Array) -> chex.Array:
    """Compute the loss for a batch of observations and their corresponding states."""
    named_logits = model(named_inputs)
    assert isinstance(named_logits, NamedArray)
    logits = named_logits.data_array
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(losses)


def validation_step(model: Layer, named_inputs: NamedArray, labels: jax.Array) -> dict[str, jax.Array]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """
    losses = loss_fn(model, named_inputs, labels)
    mean_loss = jnp.mean(losses)
    return {"loss": mean_loss}


def validate(
    model: Layer,
    cfg: ValidateConfig,
    data_generator: GenerativeProcess,
    logger: Logger | None = None,
) -> dict[str, jax.Array]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(cfg.seed)

    gen_state = data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], cfg.batch_size, axis=0)
    metrics = defaultdict(lambda: jnp.array(0.0))

    for step in range(1, cfg.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, named_inputs, labels = generate_data_batch(
            gen_states,
            data_generator,
            cfg.batch_size,
            cfg.sequence_len,
            gen_key,
        )
        metrics = validation_step(model, named_inputs, labels)
        for k, v in metrics.items():
            metrics[k] += v
        if logger and step % cfg.log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / cfg.num_steps for k, v in metrics.items()}
