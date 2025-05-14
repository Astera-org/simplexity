from collections import defaultdict

import jax
import jax.numpy as jnp
from penzai import pz
from penzai.core.named_axes import NamedArray
from penzai.nn.layer import Layer

from simplexity.configs.evaluation.config import Config
from simplexity.evaluation.metric_functions import accuracy_fn, loss_fn
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.generative_processes.state_sampler import StateSampler
from simplexity.logging.logger import Logger


def evaluation_step(model: Layer, inputs: jax.Array, labels: jax.Array) -> dict[str, jax.Array]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """
    named_inputs = pz.nx.wrap(inputs, "batch", "seq")
    named_logits = model(named_inputs)
    assert isinstance(named_logits, NamedArray)
    logits = named_logits.unwrap("batch", "seq", "vocabulary")
    token_losses = loss_fn(logits, labels)
    mean_batch_loss = jnp.mean(token_losses)
    token_accuracies = accuracy_fn(logits, labels)
    mean_batch_accuracy = jnp.mean(token_accuracies)
    metrics = {"loss": mean_batch_loss, "accuracy": mean_batch_accuracy}
    for i in range(token_losses.shape[1]):
        metrics[f"token_loss_{i}"] = jnp.mean(token_losses[:, i])
    return metrics


def evaluate(
    model: Layer,
    cfg: Config,
    data_generator: GenerativeProcess,
    state_sampler: StateSampler | None = None,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(cfg.seed)

    gen_state = data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], cfg.batch_size, axis=0)
    if state_sampler:
        sample_states = eqx.filter_jit(eqx.filter_vmap(state_sampler.sample))
    else:

        def sample_states(keys: jax.Array) -> jax.Array:
            return gen_states

    metrics = defaultdict(lambda: jnp.array(0.0))

    for step in range(1, cfg.num_steps + 1):
        key, state_key, gen_key = jax.random.split(key, 3)
        if state_sampler:
            state_keys = jax.random.split(state_key, cfg.batch_size)
            gen_states = sample_states(state_keys)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            data_generator,
            cfg.batch_size,
            cfg.sequence_len,
            gen_key,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        step_metrics = evaluation_step(model, inputs, labels)
        for metric_name, metric_value in step_metrics.items():
            metrics[metric_name] += metric_value
        if logger and step % cfg.log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / cfg.num_steps for k, v in metrics.items()}
