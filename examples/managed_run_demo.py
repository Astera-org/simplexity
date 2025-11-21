"""Managed run demo."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import logging
import logging.config
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import torch
import yaml
from torch.nn import Module as PytorchModel

import simplexity
from examples.configs.demo_config import Config
from simplexity.generative_processes.generator import generate_data_batch
from simplexity.persistence.mlflow_persister import MLFlowPersister

DEMO_DIR = Path(__file__).parent
SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


def configure_logging() -> None:
    """Load the logging configuration for the demo."""
    config_path = DEMO_DIR / "configs" / "logging.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            logging_cfg = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_cfg)
    else:
        logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=str(DEMO_DIR / "configs"), config_name="demo_config.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=True)
def main(cfg: Config, components: simplexity.Components) -> None:
    """Test the managed run decorator."""
    assert components.loggers is not None
    assert components.generative_processes is not None
    assert components.persisters is not None
    assert components.predictive_models is not None
    assert components.optimizers is not None
    is_mlflow_persister = cfg.persistence.name == "mlflow_persister"
    if is_mlflow_persister:
        persister = components.get_persister()
        assert isinstance(persister, MLFlowPersister)
        for model in components.predictive_models.values():
            if isinstance(model, PytorchModel):
                inputs = None
                if components.generative_processes:
                    generative_process = next(iter(components.generative_processes.values()))
                    batch_size = 1
                    sequence_len = 1
                    initial_state = jnp.repeat(generative_process.initial_state[None, :], batch_size, axis=0)
                    _, inputs, _ = generate_data_batch(
                        initial_state,
                        generative_process,
                        batch_size,
                        sequence_len,
                        jax.random.key(cfg.seed),
                        bos_token=cfg.generative_process.bos_token,
                        eos_token=cfg.generative_process.eos_token,
                        to_torch=True,
                    )
                    assert isinstance(inputs, torch.Tensor)
                persister.save_model_to_registry(model, "test_model", model_inputs=inputs)
            else:
                persister.save_weights(model, 0)


if __name__ == "__main__":
    configure_logging()
    main()
