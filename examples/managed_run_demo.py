from pathlib import Path

import hydra
import mlflow.pytorch as mlflow_pytorch
from torch.nn import Module as PytorchModel

import simplexity
from examples.configs.demo_config import Config
from simplexity.run_management.run_management import Components


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="demo_config.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=True)
def main(cfg: Config, components: Components) -> None:
    """Test the managed run decorator."""
    if components.logger:
        print(f"Logger: {components.logger.__class__.__name__}")
    else:
        print("No logger found")
    if components.generative_process:
        print(f"Generative process: {components.generative_process.__class__.__name__}")
    else:
        print("No generative process found")
    if components.persister:
        print(f"Persister: {components.persister.__class__.__name__}")
        framework = getattr(components.persister, "model_framework", None)
        if framework:
            print(f"Model framework: {framework}")
        else:
            print("No model framework found")
    else:
        print("No persister found")
    if components.predictive_model:
        print(f"Predictive model: {components.predictive_model.__class__.__name__}")
        is_mlflow_persister = cfg.persistence.name == "mlflow_persister"
        if is_mlflow_persister:
            instance_config = getattr(cfg.persistence, "instance", None)
            if instance_config:
                registered_model_name = getattr(instance_config, "registered_model_name", None)
                is_pytorch_model = isinstance(components.predictive_model, PytorchModel)
                if is_pytorch_model and registered_model_name:
                    mlflow_pytorch.log_model(
                        components.predictive_model,
                        name="demo",
                        registered_model_name=registered_model_name,
                    )
    else:
        print("No predictive model found")


if __name__ == "__main__":
    main()
