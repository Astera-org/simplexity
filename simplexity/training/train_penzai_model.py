import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from penzai import pz
from penzai.core.named_axes import NamedArray
from penzai.nn.layer import Layer as PenzaiModel
from penzai.toolshed import basic_training
from penzai.toolshed.basic_training import InternalTrainerState

from simplexity.configs.evaluation.config import Config as ValidateConfig
from simplexity.configs.training.config import Config as TrainConfig
from simplexity.evaluation.evaluate_penzai_model import evaluate
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.utils.hydra import typed_instantiate


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
    gen_states, obs = data_generator.generate(gen_states, batch_keys, sequence_len, False, False)
    inputs = obs[:, :-1]
    labels = obs[:, 1:]
    return gen_states, inputs, labels


def loss_fn(
    model: PenzaiModel,
    state: InternalTrainerState | None,
    rng: chex.PRNGKey,
    inputs: jax.Array,
    labels: jax.Array,
    **kwargs,
) -> tuple[jax.Array, InternalTrainerState | None, dict[str, jax.Array]]:
    """Cross entropy loss for a penzai model.

    https://penzai.readthedocs.io/en/v0.2.1/_autosummary/leaf/penzai.toolshed.basic_training.LossFunction.html
    """
    named_inputs = pz.nx.wrap(inputs, "batch", "seq")
    named_logits = model(named_inputs)
    assert isinstance(named_logits, NamedArray)
    logits = named_logits.unwrap("batch", "seq", "vocabulary")
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = jnp.mean(losses)
    return loss, state, {"loss": loss}


def train(
    model: PenzaiModel,
    training_cfg: TrainConfig,
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    validation_cfg: ValidateConfig | None = None,
    validation_data_generator: GenerativeProcess | None = None,
    persister: ModelPersister | None = None,
) -> tuple[PenzaiModel, float]:
    """Train a predictive model on a generative process."""
    key = jax.random.PRNGKey(training_cfg.seed)

    key, trainer_key = jax.random.split(key)
    optimizer = typed_instantiate(training_cfg.optimizer.instance, optax.GradientTransformation)
    trainer = basic_training.StatefulTrainer.build(
        root_rng=trainer_key,
        model=model,
        optimizer_def=optimizer,
        loss_fn=loss_fn,
    )

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
        )
        metrics = trainer.step(inputs=inputs, labels=labels)
        if logger:
            if step % training_cfg.log_every == 0:
                logger.log_metrics(step, metrics)
            if validation_cfg and validation_data_generator and step % training_cfg.validate_every == 0:
                validation_metrics = evaluate(model, validation_cfg, validation_data_generator)
                validation_metrics = {f"validation/{k}": v for k, v in validation_metrics.items()}
                logger.log_metrics(step, validation_metrics)
        if persister and step % training_cfg.checkpoint_every == 0:
            persister.save_weights(model, step)

    loss = float(metrics["loss"])
    return model, loss
