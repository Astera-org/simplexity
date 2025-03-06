import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.predictive_models.predictive_model import PredictiveModel


class TrainingState(eqx.Module):
    """State for training a model for one epoch."""

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
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
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
def training_epoch(
    state: TrainingState, attrs: TrainingAttributes, key: chex.PRNGKey
) -> tuple[TrainingState, chex.Array]:
    """Train the model for one epoch."""
    batch_keys = jax.random.split(key, attrs.batch_size)
    gen_process_states, obs = attrs.gen_process.generate(state.gen_process_states, batch_keys, attrs.sequence_len)
    state = dataclasses.replace(state, gen_process_states=gen_process_states)
    obs: jax.Array
    if obs.ndim == 2:
        obs = obs[:, :, None]
    x = obs[:, :-1, :]
    y = obs[:, 1:, :].squeeze()
    return update(state, x, y, attrs.opt_update)


@eqx.filter_jit
def train(
    key: chex.PRNGKey,
    model: PredictiveModel,
    optimizer: optax.GradientTransformation,
    gen_process: GenerativeProcess,
    initial_gen_process_state: jax.Array,
    num_epochs: int,
    batch_size: int,
    sequence_len: int,
    log_every: int = 1,
) -> tuple[PredictiveModel, jax.Array]:
    """Train a predictive model on a generative process."""
    gen_process_states = jnp.repeat(initial_gen_process_state[None, :], batch_size, axis=0)

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
        batch_size=batch_size,
        sequence_len=sequence_len,
    )

    losses = jnp.zeros(num_epochs // log_every)

    def training_loop(
        i, carry: tuple[TrainingState, jax.Array, chex.PRNGKey]
    ) -> tuple[TrainingState, jax.Array, chex.PRNGKey]:
        state, losses, key = carry
        key, epoch_key = jax.random.split(key)
        state, loss = training_epoch(state, attrs, epoch_key)
        losses = losses.at[i // log_every].set(loss)
        return state, losses, key

    state, losses, key = jax.lax.fori_loop(0, num_epochs, training_loop, (state, losses, key))

    return model, losses
