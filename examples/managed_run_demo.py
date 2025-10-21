from pathlib import Path

import hydra
from omegaconf import DictConfig

import simplexity


@hydra.main(config_path=str(Path(__file__).parent / "configs"), config_name="demo_config.yaml", version_base="1.2")
@simplexity.managed_run
def main(cfg: DictConfig) -> None:
    """Test the managed run decorator."""
    print(f"Config: {cfg}")


if __name__ == "__main__":
    main()
