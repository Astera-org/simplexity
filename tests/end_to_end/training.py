"""Managed run demo."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import mlflow
import torch
from torch.optim import Adam
from transformer_lens import HookedTransformer

import simplexity
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.persistence.mlflow_persister import MLFlowPersister
from tests.end_to_end.configs.config import Config

CONFIG_DIR = str(Path(__file__).parent / "configs")
CONFIG_NAME = "config.yaml"


@simplexity.managed_run(strict=False, verbose=True)
def train(cfg: Config, components: simplexity.Components) -> None:
    """Test the managed run decorator."""
    active_run = mlflow.active_run()
    assert active_run is not None
    logger = components.get_logger()
    assert isinstance(logger, MLFlowLogger)
    generative_process = components.get_generative_process()
    assert isinstance(generative_process, HiddenMarkovModel)
    persister = components.get_persister()
    assert isinstance(persister, MLFlowPersister)
    predictive_model = components.get_predictive_model()
    assert isinstance(predictive_model, HookedTransformer)
    optimizer = components.get_optimizer()
    assert isinstance(optimizer, Adam)

    gen_states = jnp.repeat(generative_process.initial_state[None, :], cfg.training.batch_size, axis=0)

    def generate(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = jax.random.key(step)
        _, inputs, labels = generate_data_batch(
            gen_states, generative_process, cfg.training.batch_size, cfg.training.sequence_len, key
        )
        return inputs, labels

    loss_fn = torch.nn.CrossEntropyLoss()

    def get_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return loss_fn(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1).long())

    def train_step(step: int) -> float:
        predictive_model.train()
        inputs, labels = generate(step)
        outputs = predictive_model(inputs)
        loss = get_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def log_step(step: int, loss: float) -> None:
        logger.log_metrics(step, {"loss": loss})

    eval_inputs, eval_labels = generate(cfg.training.num_steps)

    def evaluate() -> float:
        predictive_model.eval()
        outputs = predictive_model(eval_inputs)
        loss = get_loss(outputs, eval_labels)
        return loss.item()

    def eval_step(step: int) -> None:
        loss = evaluate()
        logger.log_metrics(step, {"eval_loss": loss})

    def checkpoint_step(step: int) -> None:
        persister.save_weights(predictive_model, step)

    for step in range(cfg.training.num_steps + 1):
        loss = evaluate() if step == 0 else train_step(step)
        if step % cfg.training.log_every == 0:
            log_step(step, loss)
        if step % cfg.training.evaluate_every == 0:
            eval_step(step)
        if step % cfg.training.checkpoint_every == 0:
            checkpoint_step(step)

    registered_model_name = cfg.predictive_model.name or "test_model"
    sample_inputs = generate(0)[0]
    persister.save_model_to_registry(predictive_model, registered_model_name, model_inputs=sample_inputs)


if __name__ == "__main__":
    main = hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base="1.2")(train)
    main()
