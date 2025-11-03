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
from simplexity.run_management.run_management import Components
from simplexity.utils.pip_utils import create_requirements_file

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


def configure_logging() -> None:
    """Load the logging configuration for the demo."""
    config_path = Path(__file__).parent / "configs" / "logging.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            logging_cfg = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_cfg)
    else:
        logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="demo_config.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=True)
def main(cfg: Config, components: Components) -> None:
    """Test the managed run decorator."""
    assert components.loggers is not None
    assert components.generative_processes is not None
    assert components.persisters is not None
    assert components.predictive_models is not None
    assert components.optimizer is not None
    is_mlflow_persister = cfg.persistence.name == "mlflow_persister"
    if is_mlflow_persister:
        for model in components.predictive_models:
            if isinstance(model, PytorchModel):
                timestamp = int(time.time())
                kwargs = {
                    "pytorch_model": model,
                    "name": f"demo_{timestamp}",
                    "registered_model_name": f"demo_model_{timestamp}",
                    "pip_requirements": create_requirements_file(),
                }
                if components.generative_processes and components.initial_states is not None:
                    _, inputs, _ = generate_data_batch(
                        components.initial_states[0],
                        components.generative_processes[0],
                        cfg.training.batch_size,
                        cfg.training.sequence_len,
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


if __name__ == "__main__":
    configure_logging()
    main()
