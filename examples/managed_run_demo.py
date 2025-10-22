from pathlib import Path

import hydra
from omegaconf import DictConfig

import simplexity
from simplexity.run_management.run_management import Components


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="demo_config.yaml", version_base="1.2")
@simplexity.managed_run(strict=False, verbose=True)
def main(cfg: DictConfig, components: Components) -> None:
    """Test the managed run decorator."""
    if components.logger:
        print(f"Logger: {components.logger.__class__.__name__}")
    else:
        print("No logger found")
    if components.persister:
        print(f"Persister: {components.persister.__class__.__name__}")
        framework = getattr(components.persister, "model_framework", None)
        if framework:
            print(f"Model framework: {framework}")
        else:
            print("No model framework found")
    else:
        print("No persister found")


if __name__ == "__main__":
    main()
