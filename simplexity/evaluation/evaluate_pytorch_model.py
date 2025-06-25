from collections import defaultdict
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from simplexity.configs.evaluation.config import Config
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
    cfg: Config,
    data_generator: GenerativeProcess,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Evaluate a PyTorch model on a generative process."""
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


# Alias for consistency with test expectations
evaluate_model = evaluate
