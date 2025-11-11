from collections import defaultdict
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.utils.pytorch_utils import torch_to_jax

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


def evaluation_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    metric_keys: Iterable[str] = ("loss", "accuracy"),
) -> dict[str, jax.Array]:
    """Compute evaluation metrics for a PyTorch model."""
    model.eval()
    with torch.no_grad():
        logits: torch.Tensor = model(inputs)
        metrics = {}

        # Reshape for sequence-level predictions: [batch, seq, vocab] -> [batch*seq, vocab]
        # and labels: [batch, seq] -> [batch*seq]
        vocab_size = logits.shape[2]
        logits_reshaped = logits.view(-1, vocab_size)
        labels_reshaped = labels.view(-1).long()  # Ensure labels are long type for cross entropy

        for metric_key in metric_keys:
            if metric_key == "loss":
                loss = F.cross_entropy(logits_reshaped, labels_reshaped)
                metrics[metric_key] = torch_to_jax(loss)
            elif metric_key == "accuracy":
                preds = torch.argmax(logits_reshaped, dim=-1)
                accuracy = (preds == labels_reshaped).float().mean()
                metrics[metric_key] = torch_to_jax(accuracy)

    return metrics


def evaluate(
    model: torch.nn.Module,
    cfg: DictConfig,
    data_generator: GenerativeProcess,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Evaluate a PyTorch model on a generative process."""
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
