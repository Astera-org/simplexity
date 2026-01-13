#!/usr/bin/env python
"""Compute optimal loss (entropy rate) for a generative process.

Usage:
    python -m simplexity.compute_optimal_loss <config_path> [--num-steps N] [--seed S]

Examples:
    # From a Hydra config file
    python -m simplexity.compute_optimal_loss configs/generative_process/mess3.yaml

    # With custom simulation parameters
    python -m simplexity.compute_optimal_loss configs/generative_process/unified_chain.yaml --num-steps 100000
"""

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import OmegaConf

from simplexity.generative_processes.entropy_rate import compute_entropy_rate


def load_process_from_config(config_path: str):
    """Load a generative process from a Hydra config file."""
    path = Path(config_path).resolve()

    # Load YAML directly and resolve what we can
    cfg = OmegaConf.load(path)

    # Set default values for common unresolved interpolations
    OmegaConf.register_new_resolver("device", lambda: None, replace=True)

    # Resolve the config, allowing missing values
    if "instance" in cfg:
        instance_cfg = cfg.instance
        # Remove unresolvable interpolations by setting them to None
        instance_cfg = OmegaConf.to_container(instance_cfg, resolve=False)
        if isinstance(instance_cfg, dict):
            for key, value in list(instance_cfg.items()):
                if isinstance(value, str) and value.startswith("${"):
                    instance_cfg[key] = None
            instance_cfg = OmegaConf.create(instance_cfg)
        process = instantiate(instance_cfg)
    else:
        process = instantiate(cfg)

    return process, cfg


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute optimal loss (entropy rate) for a generative process."
    )
    parser.add_argument("config_path", help="Path to the generative process config file")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50000,
        help="Number of simulation steps (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print additional information",
    )

    args = parser.parse_args()

    try:
        process, cfg = load_process_from_config(args.config_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        name = cfg.get("name", Path(args.config_path).stem)
        print(f"Process: {name}")
        print(f"Vocab size: {process.vocab_size}")
        print(f"Simulation steps: {args.num_steps}")
        print(f"Seed: {args.seed}")
        print("-" * 40)

    entropy_nats = compute_entropy_rate(process, num_steps=args.num_steps, seed=args.seed)
    entropy_bits = entropy_nats / float(jnp.log(2))
    uniform_baseline = float(jnp.log(process.vocab_size))

    if args.verbose:
        print(f"Uniform baseline: {uniform_baseline:.6f} nats")
        print(f"Optimal loss:     {entropy_nats:.6f} nats ({entropy_bits:.6f} bits)")
    else:
        print(f"{entropy_nats:.6f}")


if __name__ == "__main__":
    main()
