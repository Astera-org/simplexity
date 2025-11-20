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
import time
from pathlib import Path

import hydra
import jax
import mlflow.pytorch as mlflow_pytorch
import torch
import yaml
from mlflow.models.signature import infer_signature
from torch.nn import Module as PytorchModel

import simplexity
from examples.configs.demo_config import Config
from simplexity.generative_processes.torch_generator import generate_data_batch
from simplexity.utils.pip_utils import create_requirements_file

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
        for model in components.predictive_models.values():
            persister = components.get_persister()
            if persister:
                persister.save_weights(model, 0)
            if isinstance(model, PytorchModel):
                timestamp = int(time.time())
                kwargs = {
                    "pytorch_model": model,
                    "name": f"demo_{timestamp}",
                    "registered_model_name": f"demo_model_{timestamp}",
                    "pip_requirements": create_requirements_file(),
                }
                if components.generative_processes and components.initial_states is not None:
                    # Get the first generative process and corresponding initial state (keys match)
                    first_key = next(iter(components.generative_processes.keys()))
                    batch_size = (
                        cfg.generative_process.batch_size if cfg.generative_process.batch_size is not None else 1
                    )
                    sequence_len = (
                        cfg.generative_process.sequence_len if cfg.generative_process.sequence_len is not None else 1
                    )
                    _, inputs, _ = generate_data_batch(
                        components.initial_states[first_key],
                        components.generative_processes[first_key],
                        batch_size,
                        sequence_len,
                        jax.random.key(cfg.seed),
                        bos_token=cfg.generative_process.bos_token,
                        eos_token=cfg.generative_process.eos_token,
                    )
                    outputs: torch.Tensor = model(inputs)
                    signature = infer_signature(
                        model_input=inputs.detach().cpu().numpy(),
                        model_output=outputs.detach().cpu().numpy(),
                    )
                    kwargs["signature"] = signature
                mlflow_pytorch.log_model(**kwargs)
    
    # Demonstrate metric tracker usage
    if components.metric_trackers:
        SIMPLEXITY_LOGGER.info("[demo] demonstrating metric tracker")
        metric_tracker = components.get_metric_tracker()
        logger = components.get_logger()
        
        # Simulate a simple training loop
        model = components.get_predictive_model()
        optimizer = components.get_optimizer()
        
        if model and optimizer and components.generative_processes and components.initial_states:
            first_key = next(iter(components.generative_processes.keys()))
            batch_size = (
                cfg.generative_process.batch_size if cfg.generative_process.batch_size is not None else 4
            )
            sequence_len = (
                cfg.generative_process.sequence_len if cfg.generative_process.sequence_len is not None else 10
            )
            
            SIMPLEXITY_LOGGER.info("[demo] running 10 training steps to demonstrate metric tracking")
            for step in range(10):
                # Generate a batch of data
                _, inputs, targets = generate_data_batch(
                    components.initial_states[first_key],
                    components.generative_processes[first_key],
                    batch_size,
                    sequence_len,
                    jax.random.key(cfg.seed + step),
                    bos_token=cfg.generative_process.bos_token,
                    eos_token=cfg.generative_process.eos_token,
                )
                
                # Forward pass
                outputs = model(inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metric tracker
                metric_tracker.step(tokens=inputs.numel(), loss=loss.item())
                
                # Update metrics at different frequencies
                if step % 2 == 0:  # Update some metrics every 2 steps
                    metric_tracker.update_metrics(group="all")
                
                # Log metrics every 5 steps
                if step % 5 == 0:
                    metrics = metric_tracker.metrics()
                    SIMPLEXITY_LOGGER.info(f"[demo] Step {step} metrics: {metrics}")
                    if logger:
                        logger.log_metrics(metrics, step=step)
            
            SIMPLEXITY_LOGGER.info("[demo] metric tracking demonstration complete")



if __name__ == "__main__":
    configure_logging()
    main()
