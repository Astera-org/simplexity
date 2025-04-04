import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.validation.config import Config as ValidationConfig
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.hydra import typed_instantiate
from simplexity.validation.validate import validate


class State(eqx.Module):
    """State that evolves over the training process."""

    model: PredictiveModel
    gen_states: jax.Array
    opt_state: optax.OptState


class Attributes(eqx.Module):
    """Attributes for training."""

    data_generator: GenerativeProcess
    opt_update: optax.TransformUpdateFn
    batch_size: int
    sequence_len: int


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate a batch of data."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, obs = data_generator.generate(gen_states, batch_keys, sequence_len, False)
    inputs = jax.nn.one_hot(obs[:, :-1], data_generator.vocab_size)
    labels = obs[:, 1:]
    return gen_states, inputs, labels


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, 0, 0))
@eqx.filter_value_and_grad
def loss_fn(model: PredictiveModel, inputs: jax.Array, labels: jax.Array) -> chex.Array:
    """Compute the loss for a batch of observations and their corresponding states."""
    logits = model(inputs)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(losses)


@eqx.filter_jit
def update_model(
    state: State,
    grads: jax.Array,
    opt_update: optax.TransformUpdateFn,
) -> State:
    """Update the model parameters."""
    mean_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    params = eqx.filter(state.model, eqx.is_array)
    updates, opt_state = opt_update(mean_grads, state.opt_state, params)
    model = eqx.apply_updates(state.model, updates)
    return dataclasses.replace(state, model=model, opt_state=opt_state)


@eqx.filter_jit
def training_step(
    state: State,
    attrs: Attributes,
    inputs: jax.Array,
    labels: jax.Array,
) -> tuple[State, dict[str, jax.Array]]:
    """Train the model for one step."""
    losses, grads = loss_fn(state.model, inputs, labels)
    state = update_model(state, grads, attrs.opt_update)
    mean_loss = jnp.mean(losses)
    metrics = {"loss": mean_loss}
    return state, metrics


def train(
    model: PredictiveModel,
    training_cfg: TrainingConfig,
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    validation_cfg: ValidationConfig | None = None,
    validation_data_generator: GenerativeProcess | None = None,
    persister: ModelPersister | None = None,
) -> tuple[PredictiveModel, float]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(training_cfg.seed)

    optimizer = typed_instantiate(training_cfg.optimizer.instance, optax.GradientTransformation)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    opt_update = eqx.filter_jit(optimizer.update)

    gen_state = training_data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], training_cfg.batch_size, axis=0)
    state = State(
        model=model,
        gen_states=gen_states,
        opt_state=opt_state,
    )
    attrs = Attributes(
        data_generator=training_data_generator,
        opt_update=opt_update,
        batch_size=training_cfg.batch_size,
        sequence_len=training_cfg.sequence_len,
    )
    max_steps_digits = len(str(training_cfg.num_steps))
    metrics = {"loss": jnp.array(0.0)}

    for step in range(1, training_cfg.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            training_data_generator,
            training_cfg.batch_size,
            training_cfg.sequence_len,
            gen_key,
        )
        state, metrics = training_step(state, attrs, inputs, labels)
        if logger:
            if step % training_cfg.log_every == 0:
                logger.log_metrics(step, metrics)
            if validation_cfg and validation_data_generator and step % training_cfg.validate_every == 0:
                validation_metrics = validate(model, validation_cfg, validation_data_generator)
                validation_metrics = {f"validation/{k}": v for k, v in validation_metrics.items()}
                logger.log_metrics(step, validation_metrics)
        if persister and step % training_cfg.checkpoint_every == 0:
            full_checkpoint_name = f"{training_cfg.checkpoint_name}_{step:0{max_steps_digits}d}"
            persister.save_weights(model, full_checkpoint_name)

    loss = float(metrics["loss"])
    return model, loss
