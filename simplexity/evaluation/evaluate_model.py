from collections import defaultdict
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from simplexity.configs.evaluation.config import Config
from simplexity.evaluation.metric_functions import METRIC_FUNCTIONS
from simplexity.generative_processes.generative_process import GenerativeProtocol
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.predictive_models.predictive_model import PredictiveModel


def evaluation_step(
    model: PredictiveModel, inputs: jax.Array, labels: jax.Array, metric_keys: Iterable[str] = ("loss", "accuracy")
) -> dict[str, jax.Array]:
    """Cross entropy loss for a penzai model."""
    logits = model(inputs)
    metrics = {}
    for metric_key in metric_keys:
        metric_values = METRIC_FUNCTIONS[metric_key](logits, labels)
        metrics[metric_key] = jnp.mean(metric_values)
    return metrics


def evaluate(
    model: PredictiveModel,
    cfg: Config,
    data_generator: GenerativeProtocol,
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
        step_metrics = evaluation_step(model, inputs, labels)
        for metric_name, metric_value in step_metrics.items():
            metrics[metric_name] += metric_value
        if logger and cfg.log_every and step % cfg.log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / cfg.num_steps for k, v in metrics.items()}
