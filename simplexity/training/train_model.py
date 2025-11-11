import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig, OmegaConf
from penzai.nn.layer import Layer as PenzaiModel
from penzai.toolshed.basic_training import InternalTrainerState
from penzai.toolshed.basic_training import StatefulTrainer as PenzaiTrainer

from simplexity.evaluation.evaluate_model import evaluate
from simplexity.evaluation.metric_functions import cross_entropy_fn
from simplexity.exceptions import ConfigValidationError
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.logging.logger import Logger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.training.equinox_trainer import EquinoxTrainer
from simplexity.utils.config_utils import typed_instantiate
from simplexity.utils.equinox import vmap_model
from simplexity.utils.penzai import use_penzai_model


def loss_fn(
    model: PredictiveModel,
    state: InternalTrainerState | None,
    rng: chex.PRNGKey,
    inputs: jax.Array,
    labels: jax.Array,
    **kwargs,
) -> tuple[jax.Array, InternalTrainerState | None, dict[str, jax.Array]]:
    """Cross entropy loss."""
    logits = model(inputs)
    losses = cross_entropy_fn(logits, labels)
    loss = jnp.mean(losses)
    return loss, state, {"loss": loss}


def train(
    model: PredictiveModel,
    training_cfg: DictConfig,
    training_data_generator: GenerativeProcess,
    logger: Logger | None = None,
    validation_cfg: DictConfig | None = None,
    validation_data_generator: GenerativeProcess | None = None,
    persister: ModelPersister | None = None,
    training_bos_token: int | None = None,
    training_eos_token: int | None = None,
    validation_bos_token: int | None = None,
    validation_eos_token: int | None = None,
) -> tuple[PredictiveModel, float]:
    """Train a predictive model on a generative process."""
    seed = training_cfg.get("seed", 0)
    batch_size: int | None = training_cfg.get("batch_size", None)
    if batch_size is None:
        raise ConfigValidationError("batch_size is required")
    sequence_len: int | None = training_cfg.get("sequence_len", None)
    if sequence_len is None:
        raise ConfigValidationError("sequence_len is required")
    num_steps: int | None = training_cfg.get("num_steps", None)
    if num_steps is None:
        raise ConfigValidationError("num_steps is required")
    log_every: int | None = training_cfg.get("log_every", None)
    validate_every: int | None = training_cfg.get("validate_every", None)
    checkpoint_every: int | None = training_cfg.get("checkpoint_every", None)
    key = jax.random.PRNGKey(seed)

    def get_model():
        return model

    key, trainer_key = jax.random.split(key)
    optimizer_instance_config = OmegaConf.select(training_cfg, "optimizer.instance")
    optimizer = typed_instantiate(optimizer_instance_config, optax.GradientTransformation)
    if isinstance(model, PenzaiModel):
        trainer = PenzaiTrainer.build(
            root_rng=trainer_key,
            model=model,
            optimizer_def=optimizer,
            loss_fn=use_penzai_model(loss_fn),
        )

        validate = use_penzai_model(evaluate)
    elif isinstance(model, eqx.Module):
        trainer = EquinoxTrainer.build(
            root_rng=trainer_key,
            model=model,
            optimizer_def=optimizer,
            loss_fn=vmap_model(loss_fn),
        )

        def get_trainer_model():
            return trainer.model.value

        get_model = get_trainer_model

        validate = vmap_model(evaluate)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    gen_state = training_data_generator.initial_state
    gen_states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    metrics = {"loss": jnp.array(0.0)}

    for step in range(1, num_steps + 1):
        key, gen_key = jax.random.split(key)
        gen_states, inputs, labels = generate_data_batch(
            gen_states,
            training_data_generator,
            batch_size,
            sequence_len,
            gen_key,
            bos_token=training_bos_token,
            eos_token=training_eos_token,
        )
        metrics = trainer.step(inputs=inputs, labels=labels)
        if logger:
            if log_every and step % log_every == 0:
                logger.log_metrics(step, metrics)
            if validation_cfg and validation_data_generator and validate_every and step % validate_every == 0:
                validation_metrics = validate(
                    model=get_model(),
                    cfg=validation_cfg,
                    data_generator=validation_data_generator,
                    bos_token=validation_bos_token,
                    eos_token=validation_eos_token,
                )
                validation_metrics = {f"validation/{k}": v for k, v in validation_metrics.items()}
                logger.log_metrics(step, validation_metrics)
        if persister and checkpoint_every and step % checkpoint_every == 0:
            persister.save_weights(get_model(), step)

    loss = float(metrics["loss"])
    return get_model(), loss
