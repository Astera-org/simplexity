from collections import defaultdict
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from simplexity.evaluation.metric_functions import METRIC_FUNCTIONS
from simplexity.exceptions import ConfigValidationError
from simplexity.generative_processes.generative_process import GenerativeProcess
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
    cfg: DictConfig,
    data_generator: GenerativeProcess,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Train a predictive model on a generative process."""
    seed: int = cfg.get("seed", 0)
    batch_size: int | None = cfg.get("batch_size", None)
    if batch_size is None:
        raise ConfigValidationError("batch_size is required")
    sequence_len: int | None = cfg.get("sequence_len", None)
    if sequence_len is None:
        raise ConfigValidationError("sequence_len is required")
    num_steps: int | None = cfg.get("num_steps", None)
    if num_steps is None:
        raise ConfigValidationError("num_steps is required")
    log_every: int | None = cfg.get("log_every", None)

    key = jax.random.PRNGKey(seed)

    gen_state = data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    metrics = defaultdict(lambda: jnp.array(0.0))

    for step in range(1, num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            data_generator,
            batch_size,
            sequence_len,
            gen_key,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        step_metrics = evaluation_step(model, inputs, labels)
        for metric_name, metric_value in step_metrics.items():
            metrics[metric_name] += metric_value
        if logger and log_every and step % log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / num_steps for k, v in metrics.items()}
