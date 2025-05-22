import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from simplexity.configs.evaluation.config import Config as ValidationConfig
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.evaluation.evaluate_equinox_model import evaluate
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.hydra import typed_instantiate


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
    model: PredictiveModel,
    opt_state: optax.OptState,
    grads: jax.Array,
    opt_update: optax.TransformUpdateFn,
) -> tuple[PredictiveModel, optax.OptState]:
    """Update the model parameters."""
    mean_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt_update(mean_grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


@eqx.filter_jit
def training_step(
    model: PredictiveModel,
    opt_state: optax.OptState,
    inputs: jax.Array,
    labels: jax.Array,
    opt_update: optax.TransformUpdateFn,
) -> tuple[PredictiveModel, optax.OptState, dict[str, jax.Array]]:
    """Train the model for one step."""
    losses, grads = loss_fn(model, inputs, labels)
    model, opt_state = update_model(model, opt_state, grads, opt_update)
    mean_loss = jnp.mean(losses)
    metrics = {"loss": mean_loss}
    return model, opt_state, metrics


def train(
    model: PredictiveModel,
    training_cfg: TrainingConfig,
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    validation_cfg: ValidationConfig | None = None,
    validation_data_generator: GenerativeProcess | None = None,
    persister: ModelPersister | None = None,
    training_bos_token: int | None = None,
    training_eos_token: int | None = None,
    validation_bos_token: int | None = None,
    validation_eos_token: int | None = None,
) -> tuple[PredictiveModel, float]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(training_cfg.seed)

    optimizer = typed_instantiate(training_cfg.optimizer.instance, optax.GradientTransformation)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    opt_update = eqx.filter_jit(optimizer.update)

    gen_state = training_data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], training_cfg.batch_size, axis=0)

    metrics = {"loss": jnp.array(0.0)}

    for step in range(1, training_cfg.num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            training_data_generator,
            training_cfg.batch_size,
            training_cfg.sequence_len,
            gen_key,
            bos_token=training_bos_token,
            eos_token=training_eos_token,
        )
        model, opt_state, metrics = training_step(model, opt_state, inputs, labels, opt_update)
        if logger:
            if step % training_cfg.log_every == 0:
                logger.log_metrics(step, metrics)
            if validation_cfg and validation_data_generator and step % training_cfg.validate_every == 0:
                validation_metrics = evaluate(
                    model,
                    validation_cfg,
                    validation_data_generator,
                    bos_token=validation_bos_token,
                    eos_token=validation_eos_token,
                )
                validation_metrics = {f"validation/{k}": v for k, v in validation_metrics.items()}
                logger.log_metrics(step, validation_metrics)
        if persister and step % training_cfg.checkpoint_every == 0:
            persister.save_weights(model, step)

    loss = float(metrics["loss"])
    return model, loss
