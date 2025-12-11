"""End-to-end training script for testing."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import mlflow
import torch
import tqdm
from torch.optim import Adam
from transformer_lens import HookedTransformer

import simplexity
from simplexity.generative_processes.factored_generative_process import FactoredGenerativeProcess
from simplexity.generative_processes.hidden_markov_model import GeneralizedHiddenMarkovModel, HiddenMarkovModel
from simplexity.generative_processes.torch_generator import generate_data_batch, generate_data_batch_with_full_history
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.metrics.metric_tracker import MetricTracker
from simplexity.persistence.mlflow_persister import MLFlowPersister
from simplexity.structured_configs.activation_tracker import ActivationTrackerConfig
from simplexity.structured_configs.generative_process import GenerativeProcessConfig
from simplexity.structured_configs.learning_rate_scheduler import LearningRateSchedulerConfig
from simplexity.structured_configs.logging import LoggingConfig
from simplexity.structured_configs.metric_tracker import MetricTrackerConfig
from simplexity.structured_configs.mlflow import MLFlowConfig
from simplexity.structured_configs.optimizer import OptimizerConfig
from simplexity.structured_configs.persistence import PersistenceConfig
from simplexity.structured_configs.predictive_model import PredictiveModelConfig

CONFIG_DIR = str(Path(__file__).parent / "configs")
CONFIG_NAME = "training_test.yaml"

logging.getLogger("databricks.sdk").setLevel(logging.WARNING)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_steps: int
    batch_size: int
    sequence_len: int
    log_cheap_every: int
    log_expensive_every: int
    checkpoint_every: int
    evaluate_every: int
    validation_multiplier: int


@dataclass
class TrainingRunConfig:
    """Configuration for the managed run demo."""

    mlflow: MLFlowConfig
    logging: LoggingConfig
    generative_process: GenerativeProcessConfig
    persistence: PersistenceConfig
    predictive_model: PredictiveModelConfig
    optimizer: OptimizerConfig
    learning_rate_scheduler: LearningRateSchedulerConfig
    training_metric_tracker: MetricTrackerConfig
    eval_metric_tracker: MetricTrackerConfig
    training: TrainingConfig
    activation_tracker: ActivationTrackerConfig

    device: str
    experiment_name: str
    run_name: str
    seed: int
    tags: dict[str, str]


def _expand_init_state(
    initial_state: jax.Array | tuple[jax.Array, ...],
    batch_size: int,
) -> jax.Array | tuple[jax.Array, ...]:
    """Expand the initial state to the batch size."""
    if isinstance(initial_state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in initial_state)
    return jnp.repeat(initial_state[None, :], batch_size, axis=0)


@simplexity.managed_run(strict=False, verbose=True)
def train(cfg: TrainingRunConfig, components: simplexity.Components) -> None:
    """Test the managed run decorator."""
    active_run = mlflow.active_run()
    assert active_run is not None
    logger = components.get_logger()
    assert isinstance(logger, MLFlowLogger)
    generative_process = components.get_generative_process()
    assert isinstance(generative_process, (HiddenMarkovModel, GeneralizedHiddenMarkovModel, FactoredGenerativeProcess))
    persister = components.get_persister()
    assert isinstance(persister, MLFlowPersister)
    predictive_model = components.get_predictive_model()
    assert isinstance(predictive_model, HookedTransformer)
    optimizer = components.get_optimizer()
    assert isinstance(optimizer, Adam)
    learning_rate_scheduler = components.get_learning_rate_scheduler()
    assert isinstance(learning_rate_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    training_metric_tracker = components.get_metric_tracker("training_metric_tracker")
    assert isinstance(training_metric_tracker, MetricTracker)
    eval_metric_tracker = components.get_metric_tracker("eval_metric_tracker")
    assert isinstance(eval_metric_tracker, MetricTracker)
    activation_tracker = components.get_activation_tracker()
    assert activation_tracker is not None

    visualization_path = TemporaryDirectory()

    gen_states = _expand_init_state(
        generative_process.initial_state,
        cfg.training.batch_size,
    )

    # Only need to specify device for MPS since JAX doesn't support it
    # (JAX will use CPU while PyTorch model is on MPS)
    model_device = next(predictive_model.parameters()).device
    device_arg = model_device if model_device.type == "mps" else None

    def generate(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = jax.random.key(step)
        _, inputs, labels = generate_data_batch(
            gen_states,
            generative_process,
            cfg.training.batch_size,
            cfg.training.sequence_len,
            key,
            device=device_arg,
            bos_token=cfg.generative_process.bos_token,
        )
        return inputs, labels

    loss_fn = torch.nn.CrossEntropyLoss()

    def get_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1).long())

    def train_step(step: int):
        predictive_model.train()
        inputs, labels = generate(step)
        outputs = predictive_model(inputs)
        loss = get_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_metric_tracker.step(tokens=inputs, loss=loss)

    def log_step(step: int, group: str) -> None:
        metrics = training_metric_tracker.get_metrics(group)
        logger.log_metrics(step, metrics)

    eval_inputs, eval_labels = generate(cfg.training.num_steps)

    def evaluate() -> float:
        predictive_model.eval()
        outputs = predictive_model(eval_inputs)
        loss = get_loss(outputs, eval_labels)
        return float(loss.detach().item())

    def add_key_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}/{k}": v for k, v in d.items()}

    def eval_step(step: int) -> None:
        loss = evaluate()
        eval_metric_tracker.step(loss=loss)
        metrics = eval_metric_tracker.get_metrics()
        metrics = add_key_prefix(metrics, "eval")
        logger.log_metrics(step, metrics)
        if step != 0:
            old_lr = optimizer.param_groups[0]["lr"]
            learning_rate_scheduler.step(metrics["eval/loss/step"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                logging.info(f"Learning rate changed from {old_lr} to {new_lr} at step {step}")

    def activation_tracker_step(step: int) -> None:
        predictive_model.eval()
        outs = generate_data_batch_with_full_history(
            _expand_init_state(
                generative_process.initial_state,
                cfg.training.batch_size * cfg.training.validation_multiplier,
            ),
            generative_process,
            cfg.training.batch_size * cfg.training.validation_multiplier,
            cfg.training.sequence_len,
            jax.random.key(step),
            device=device_arg,
            bos_token=cfg.generative_process.bos_token,
        )
        inputs = outs["inputs"]
        assert isinstance(inputs, (jax.Array, torch.Tensor))
        prefix_probs = outs["prefix_probabilities"]
        assert isinstance(prefix_probs, (jax.Array, torch.Tensor))
        _, act_cache = predictive_model.run_with_cache(inputs)
        act_cache = {k: v.detach().cpu() for k, v in act_cache.items() if "resid" in k}
        scalars, _, visualizations = activation_tracker.analyze(
            inputs=inputs,
            beliefs=outs["belief_states"],
            probs=prefix_probs,
            activations=act_cache,
            step=step,
        )
        visualization_paths = activation_tracker.save_visualizations(
            visualizations, Path(visualization_path.name), step
        )
        for key, path in visualization_paths.items():
            logger.log_artifact(str(path), artifact_path=f"activation_plots/{key.split('/')[0]}")
        scalars = add_key_prefix(dict(scalars), "activations")
        logger.log_metrics(step, scalars)

    def checkpoint_step(step: int) -> None:
        persister.save_weights(predictive_model, step)

    for step in tqdm.tqdm(range(cfg.training.num_steps + 1)):
        if step == 0:
            initial_loss = evaluate()
            training_metric_tracker.context.loss = initial_loss
            eval_metric_tracker.context.loss = initial_loss
            activation_tracker_step(step)
        else:
            train_step(step)
        if step % cfg.training.log_cheap_every == 0:
            log_step(step, "cheap")
        if step % cfg.training.log_expensive_every == 0:
            log_step(step, "expensive")
            activation_tracker_step(step)
        if step % cfg.training.evaluate_every == 0:
            eval_step(step)
        if step % cfg.training.checkpoint_every == 0:
            checkpoint_step(step)

    registered_model_name = cfg.predictive_model.name or "test_model"
    sample_inputs = generate(0)[0]
    # TODO(https://github.com/Astera-org/simplexity/issues/125): This is a hack
    step += 1  # pyright: ignore[reportPossiblyUnboundVariable]
    persister.save_model_to_registry(predictive_model, registered_model_name, model_inputs=sample_inputs, step=step)


if __name__ == "__main__":
    main = hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base="1.2")(train)
    main()
