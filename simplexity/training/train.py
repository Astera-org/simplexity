import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from simplexity.configs.train.config import Config as TrainConfig
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.predictive_models.predictive_model import PredictiveModel


class TrainingState(eqx.Module):
    """State for training a model for one step."""

    model: PredictiveModel
    gen_process_states: jax.Array
    opt_state: optax.OptState


class TrainingAttributes(eqx.Module):
    """Attributes for training."""

    gen_process: GenerativeProcess
    opt_update: optax.TransformUpdateFn
    batch_size: int
    sequence_len: int


@eqx.filter_vmap(in_axes=(None, 0, 0))
@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model: PredictiveModel, x: jax.Array, y: jax.Array) -> chex.Array:
    """Compute the loss for a batch of observations and their corresponding states."""
    logits = model(x)
    losses = optax.softmax_cross_entropy(logits, y)
    return jnp.mean(losses)


@eqx.filter_jit
def update(
    state: TrainingState,
    x: jax.Array,
    y: jax.Array,
    opt_update: optax.TransformUpdateFn,
) -> tuple[TrainingState, chex.Array]:
    """Update the model parameters."""
    loss, grads = loss_fn(state.model, x, y)
    mean_loss = jnp.mean(loss)
    mean_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    params = eqx.filter(state.model, eqx.is_array)
    updates, opt_state = opt_update(mean_grads, state.opt_state, params)
    model = eqx.apply_updates(state.model, updates)
    return dataclasses.replace(state, model=model, opt_state=opt_state), mean_loss


@eqx.filter_jit
def training_step(
    state: TrainingState, attrs: TrainingAttributes, key: chex.PRNGKey
) -> tuple[TrainingState, chex.Array]:
    """Train the model for one step."""
    batch_keys = jax.random.split(key, attrs.batch_size)
    gen_process_states, obs = attrs.gen_process.generate(state.gen_process_states, batch_keys, attrs.sequence_len)
    state = dataclasses.replace(state, gen_process_states=gen_process_states)
    one_hot_obs = jax.nn.one_hot(obs, state.model.out_size)
    x = one_hot_obs[:, :-1, :]
    y = one_hot_obs[:, 1:, :].squeeze()
    return update(state, x, y, attrs.opt_update)


def train(
    cfg: TrainConfig,
    model: PredictiveModel,
    gen_process: GenerativeProcess,
    initial_gen_process_state: jax.Array,
) -> tuple[PredictiveModel, jax.Array]:
    """Train a predictive model on a generative process."""
    gen_process_states = jnp.repeat(initial_gen_process_state[None, :], cfg.batch_size, axis=0)

    optimizer = typed_instantiate(cfg.optimizer.instance, optax.GradientTransformation)

    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    opt_update = eqx.filter_jit(optimizer.update)

    state = TrainingState(
        model=model,
        gen_process_states=gen_process_states,
        opt_state=opt_state,
    )
    attrs = TrainingAttributes(
        gen_process=gen_process,
        opt_update=opt_update,
        batch_size=cfg.batch_size,
        sequence_len=cfg.sequence_len,
    )

    losses = jnp.zeros(cfg.num_steps // cfg.log_every)

    key = jax.random.PRNGKey(cfg.seed)
    for i in range(1, cfg.num_steps + 1):
        key, step_key = jax.random.split(key)
        state, loss = training_step(state, attrs, step_key)
        losses = losses.at[i // cfg.log_every].set(loss)

    return model, losses
