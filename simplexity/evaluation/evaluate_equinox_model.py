from collections import defaultdict

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from simplexity.configs.evaluation.config import Config
from simplexity.evaluation.metric_functions import accuracy_fn, loss_fn
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.generative_processes.state_sampler import StateSampler
from simplexity.logging.logger import Logger
from simplexity.predictive_models.gru_rnn import GRURNN, GRUFn
from simplexity.predictive_models.predictive_model import PredictiveModel


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, 0, 0))
def evaluation_step(model: PredictiveModel, inputs: jax.Array, labels: jax.Array) -> dict[str, jax.Array]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """

    def loss_fn_with_model(model):
        logits = model(inputs)
        token_losses = loss_fn(logits, labels)
        return jnp.mean(token_losses)

    # Compute gradients
    grads = jax.grad(loss_fn_with_model)(model)

    # Compute gradient norms
    grad_norms = tree_map(lambda x: jnp.linalg.norm(x) if x is not None else 0.0, grads)
    total_grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grad_norms)))

    # Compute original metrics
    logits = model(inputs)
    token_losses = loss_fn(logits, labels)
    mean_sequence_loss = jnp.mean(token_losses)
    token_accuracies = accuracy_fn(logits, labels)
    mean_sequence_accuracy = jnp.mean(token_accuracies)

    metrics = {"loss": mean_sequence_loss, "accuracy": mean_sequence_accuracy, "grad_norm": total_grad_norm}

    # Compute weight norms
    weight_norms = tree_map(lambda x: jnp.linalg.norm(x) if x is not None else 0.0, model)
    total_weight_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(weight_norms)))
    metrics["weight_norm"] = total_weight_norm

    # Add hidden state norms for GRU layers
    if isinstance(model, GRURNN):
        hidden_states = []
        x = inputs
        for layer in model.layers:
            if isinstance(layer, eqx.nn.Lambda) and isinstance(layer.fn, GRUFn):
                # Get hidden states from GRU layer
                gru_fn = layer.fn

                def process_element(carry, x, gru_fn=gru_fn):
                    next_carry = gru_fn.cell(x, carry)
                    return next_carry, next_carry

                hidden = jnp.zeros(gru_fn.cell.hidden_size)
                _, layer_states = jax.lax.scan(process_element, hidden, x)
                hidden_states.append(layer_states)
                x = layer_states
            else:
                x = layer(x)

        # Compute norms for each layer's hidden states
        for i, states in enumerate(hidden_states):
            # Compute mean norm across sequence length and batch
            state_norm = jnp.mean(jnp.linalg.norm(states, axis=-1))
            metrics[f"gru_layer_{i}_state_norm"] = state_norm

    for i in range(min(token_losses.shape[0], 100)):
        metrics[f"token_loss_{i}"] = jnp.mean(token_losses[i])
    return metrics


def evaluate(
    model: PredictiveModel,
    cfg: Config,
    data_generator: GenerativeProcess,
    state_sampler: StateSampler | None = None,
    logger: Logger | None = None,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(cfg.seed)

    vocab_size = data_generator.vocab_size
    if bos_token:
        vocab_size += 1
    if eos_token:
        vocab_size += 1

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
        inputs = jax.nn.one_hot(inputs, vocab_size)
        step_metrics: dict[str, jax.Array] = evaluation_step(model, inputs, labels)
        for metric_name, batch_metric_values in step_metrics.items():
            mean_batch_metric_value = jnp.mean(batch_metric_values)
            metrics[metric_name] += mean_batch_metric_value
        if logger and step % cfg.log_every == 0:
            logger.log_metrics(step, metrics)

    return {k: v / cfg.num_steps for k, v in metrics.items()}
