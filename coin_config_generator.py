import argparse
from pathlib import Path
from typing import Any

import yaml


def generate_coin_config(n: int) -> dict[str, Any]:
    """Generate a YAML config for a nonergodic process composed of n coin processes."""
    return {
        "name": f"{n}_coins",
        "vocab_size": 2,
        "instance": {
            "_target_": "simplexity.generative_processes.builder.build_nonergodic_hidden_markov_model",
            "process_names": ["coin"] * n,
            "process_kwargs": [{"p": (i + 1) / (n + 1)} for i in range(n)],
            "process_weights": [1.0 / n] * n,
            "vocab_maps": [[0, 1] for _ in range(n)],
        },
    }


def save_coin_config(n: int, output_path: Path) -> None:
    """Generate and save a coin process YAML config to a file."""
    config = generate_coin_config(n)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_coin_configs(n_min: int, n_max: int, output_dir: Path) -> None:
    """Generate and save coin process YAML configs to a directory."""
    for n in range(n_min, n_max + 1):
        save_coin_config(n, output_dir / f"{n}_coins.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_min", type=int, default=1)
    parser.add_argument("--n_max", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default="simplexity/configs/generative_process")
    args = parser.parse_args()
    save_coin_configs(args.n_min, args.n_max, args.output_dir)
