import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from simplexity.configs.train.config import Config as TrainConfig
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.hydra_helpers import typed_instantiate
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel


class TrainingState(eqx.Module):
    """State for training a model for one step."""

    model: PredictiveModel
    train_gen_states: jax.Array
    val_gen_states: jax.Array
    opt_state: optax.OptState


class TrainingAttributes(eqx.Module):
    """Attributes for training."""

    train_data_generator: GenerativeProcess
    val_data_generator: GenerativeProcess
    opt_update: optax.TransformUpdateFn
    batch_size: int
    sequence_len: int


def loss_fn(model: PredictiveModel, x: jax.Array, y: jax.Array) -> chex.Array:
    """Compute the loss for a batch of observations and their corresponding states."""
    logits = model(x)
    losses = optax.softmax_cross_entropy(logits, y)
    return jnp.mean(losses)


train_loss_fn = eqx.filter_vmap(eqx.filter_jit(eqx.filter_value_and_grad(loss_fn)), in_axes=(None, 0, 0))
val_loss_fn = eqx.filter_vmap(eqx.filter_jit(loss_fn), in_axes=(None, 0, 0))


@eqx.filter_jit
def update_model(
    state: TrainingState,
    grads: jax.Array,
    opt_update: optax.TransformUpdateFn,
) -> TrainingState:
    """Update the model parameters."""
    mean_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    params = eqx.filter(state.model, eqx.is_array)
    updates, opt_state = opt_update(mean_grads, state.opt_state, params)
    model = eqx.apply_updates(state.model, updates)
    return dataclasses.replace(state, model=model, opt_state=opt_state)


@eqx.filter_jit
def step_model(
    state: TrainingState, attrs: TrainingAttributes, key: chex.PRNGKey, *, train: bool = True
) -> tuple[TrainingState, chex.Array]:
    """Train the model for one step."""
    batch_keys = jax.random.split(key, attrs.batch_size)
    if train:
        train_gen_states, obs = attrs.train_data_generator.generate(
            state.train_gen_states, batch_keys, attrs.sequence_len, False
        )
        vocab_size = attrs.train_data_generator.vocab_size
        state = dataclasses.replace(state, train_gen_states=train_gen_states)
    else:
        val_gen_states, obs = attrs.val_data_generator.generate(
            state.val_gen_states, batch_keys, attrs.sequence_len, False
        )
        vocab_size = attrs.val_data_generator.vocab_size
        state = dataclasses.replace(state, val_gen_states=val_gen_states)
    one_hot_obs = jax.nn.one_hot(obs, vocab_size)
    x = one_hot_obs[:, :-1, :]
    y = one_hot_obs[:, 1:, :].squeeze()
    if train:
        losses, grads = train_loss_fn(state.model, x, y)
        state = update_model(state, grads, attrs.opt_update)
    else:
        losses = val_loss_fn(state.model, x, y)
    mean_loss = jnp.mean(losses)
    return state, mean_loss


@eqx.filter_jit
def validate_model(
    state: TrainingState,
    attrs: TrainingAttributes,
    key: chex.PRNGKey,
    num_validation_steps: int,
) -> tuple[TrainingState, dict[str, jax.Array]]:
    """Compute the validation loss."""

    def loop_body(_, carry):
        state, key, total_loss = carry
        key, step_key = jax.random.split(key)
        state, loss = step_model(state, attrs, step_key, train=False)
        total_loss = total_loss + loss
        return state, key, total_loss

    state, _, total_loss = jax.lax.fori_loop(0, num_validation_steps, loop_body, (state, key, jnp.array(0.0)))
    mean_loss = total_loss / num_validation_steps
    metrics = {"validation_loss": mean_loss}
    return state, metrics


def train(
    cfg: TrainConfig,
    model: PredictiveModel,
    training_data_generator: GenerativeProcess,
    validation_data_generator: GenerativeProcess,
    persister: ModelPersister,
    logger: Logger,
) -> tuple[PredictiveModel, float]:
    """Train a predictive model on a generative process."""
    train_gen_state = training_data_generator.initial_state
    train_gen_states = jnp.repeat(train_gen_state[None, :], cfg.batch_size, axis=0)

    val_gen_state = validation_data_generator.initial_state
    val_gen_states = jnp.repeat(val_gen_state[None, :], cfg.batch_size, axis=0)

    optimizer = typed_instantiate(cfg.optimizer.instance, optax.GradientTransformation)

    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    opt_update = eqx.filter_jit(optimizer.update)

    state = TrainingState(
        model=model,
        train_gen_states=train_gen_states,
        val_gen_states=val_gen_states,
        opt_state=opt_state,
    )
    attrs = TrainingAttributes(
        train_data_generator=training_data_generator,
        val_data_generator=validation_data_generator,
        opt_update=opt_update,
        batch_size=cfg.batch_size,
        sequence_len=cfg.sequence_len,
    )

    key = jax.random.PRNGKey(cfg.seed)
    train_key, val_key = jax.random.split(key)
    max_steps_digits = len(str(cfg.num_steps))
    loss = jnp.array(0.0)
    for step in range(1, cfg.num_steps + 1):
        train_key, step_key = jax.random.split(train_key)
        state, loss = step_model(state, attrs, step_key, train=True)
        if step % cfg.log_every == 0:
            logger.log_metrics(step, {"loss": loss})
        if step % cfg.validate_every == 0:
            state, val_metrics = validate_model(state, attrs, val_key, cfg.num_validation_steps)
            logger.log_metrics(step, val_metrics)
        if step % cfg.checkpoint_every == 0:
            full_checkpoint_name = f"{cfg.checkpoint_name}_{step:0{max_steps_digits}d}"
            persister.save_weights(model, full_checkpoint_name)

    return model, float(loss)
