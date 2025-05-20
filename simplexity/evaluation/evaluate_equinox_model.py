from collections import defaultdict

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.configs.evaluation.config import Config
from simplexity.evaluation.metric_functions import accuracy_fn, loss_fn
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.predictive_models.predictive_model import PredictiveModel


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, 0, 0))
def evaluation_step(model: PredictiveModel, inputs: jax.Array, labels: jax.Array) -> dict[str, jax.Array]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """
    logits = model(inputs)
    token_losses = loss_fn(logits, labels)
    mean_sequence_loss = jnp.mean(token_losses)
    token_accuracies = accuracy_fn(logits, labels)
    mean_sequence_accuracy = jnp.mean(token_accuracies)
    return {"loss": mean_sequence_loss, "accuracy": mean_sequence_accuracy}


def evaluate(
    model: PredictiveModel,
    cfg: Config,
    data_generator: GenerativeProcess,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(cfg.seed)

    gen_state = data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], cfg.batch_size, axis=0)
    metrics = defaultdict(lambda: jnp.array(0.0))

    for step in range(1, cfg.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            data_generator,
            cfg.batch_size,
            cfg.sequence_len,
            gen_key,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        inputs = jax.nn.one_hot(inputs, data_generator.vocab_size)
        step_metrics: dict[str, jax.Array] = evaluation_step(model, inputs, labels)
        for metric_name, batch_metric_values in step_metrics.items():
            mean_batch_metric_value = jnp.mean(batch_metric_values)
            metrics[metric_name] += mean_batch_metric_value
        if logger and step % cfg.log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / cfg.num_steps for k, v in metrics.items()}
