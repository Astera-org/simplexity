#!/usr/bin/env python
r"""Run multiple Hydra experiments in parallel across GPUs or CPU.

simplexity-multirun is a CLI tool for running multiple Hydra experiments in
parallel with proper device isolation. It's a simpler alternative to Ray or
Hydra's joblib launcher when you just need to run experiments on a single machine.

How It Works
------------
Each experiment runs in a separate subprocess with CUDA_VISIBLE_DEVICES set to
ensure exclusive GPU access. Jobs are assigned to devices round-robin and started
with a 5-second stagger to avoid initialization race conditions.

Device Modes
------------
GPU Mode (--gpus):
    Specify which GPUs to use. Each job gets exclusive access to one GPU.
    Jobs are distributed round-robin across the specified GPUs.

    Example: --gpus 0,1 with 4 jobs -> Job0:GPU0, Job1:GPU1, Job2:GPU0, Job3:GPU1

CPU Mode (--cpu --workers N):
    Run without GPU acceleration. Specify number of parallel workers.
    Sets CUDA_VISIBLE_DEVICES="" to disable GPU for all jobs.

Sweep Modes
-----------
--sweep (Cartesian Product):
    Generate all combinations of parameters. Can specify multiple times.

    Example: --sweep 'a=1,2' --sweep 'b=x,y'
    Generates: a=1 b=x, a=1 b=y, a=2 b=x, a=2 b=y (4 jobs)

--sweep-file (Config File):
    Load sweep parameters from a YAML file. Can be combined with --sweep.

    Example file (sweeps/my_experiment.yaml):
        seed: [1, 2, 3, 4]
        model.lr: [0.01, 0.001]

    Example: --sweep-file sweeps/my_experiment.yaml
    Generates: 4 x 2 = 8 jobs (cartesian product)

--overrides (Explicit List):
    Run specific override strings. Each string is one job.

    Example: --overrides 'seed=1 lr=0.01' 'seed=2 lr=0.001'
    Generates: 2 jobs with those exact overrides

Usage Examples
--------------
    # Basic sweep across 2 GPUs
    simplexity-multirun run.py -c train_config --gpus 0,1 --sweep 'seed=1,2,3,4'

    # Load sweep params from a YAML file
    simplexity-multirun run.py -c train_config --gpus 0,1 --sweep-file sweeps/experiment.yaml

    # CPU-only mode with 4 parallel workers
    simplexity-multirun run.py -c train_config --cpu --workers 4 --sweep 'seed=1,2,3,4'

    # Cartesian product: 2x2 = 4 experiments
    simplexity-multirun run.py -c train_config --gpus 0,1,2,3 \\
        --sweep 'model.n_heads=1,2' \\
        --sweep 'model.n_layers=1,2'

    # Limit parallelism (e.g., 4 GPUs but only 2 jobs at a time)
    simplexity-multirun run.py -c train_config --gpus 0,1,2,3 --max-parallel 2 --sweep 'seed=1,2,3,4'

    # Dry run to preview commands without executing
    simplexity-multirun run.py -c train_config --gpus 0,1 --sweep 'seed=1,2' --dry-run

Why Not Ray/Joblib?
-------------------
- No package shipping or virtual environment creation
- Simple subprocess isolation - easy to debug
- No complex configuration or runtime environments
- Works reliably for single-machine multi-GPU setups
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from omegaconf import OmegaConf

# Delay between starting jobs to avoid initialization race conditions
JOB_START_DELAY_SECONDS = 5


def load_sweep_file(path: str) -> list[str]:
    """Load sweep parameters from a YAML file.

    The file should contain parameter names as keys and lists of values:

        seed: [1, 2, 3, 4]
        model.lr: [0.01, 0.001]

    Args:
        path: Path to the sweep YAML file.

    Returns:
        List of sweep strings like ['seed=1,2,3,4', 'model.lr=0.01,0.001']
    """
    cfg = OmegaConf.load(path)
    sweeps = []
    for key, values in cfg.items():
        # Convert OmegaConf types to Python types
        values = OmegaConf.to_object(values) if OmegaConf.is_config(values) else values
        if isinstance(values, (list, tuple)):
            values_str = ",".join(str(v) for v in values)
        else:
            values_str = str(values)
        sweeps.append(f"{key}={values_str}")
    return sweeps


def parse_sweep_param(sweep_str: str) -> tuple[str, list[str]]:
    """Parse a sweep parameter like 'param=1,2,3' into (param, [1, 2, 3])."""
    key, values = sweep_str.split("=", 1)
    return key, [v.strip() for v in values.split(",")]


def generate_override_combinations(sweeps: list[str]) -> list[str]:
    """Generate all combinations of sweep parameters (cartesian product).

    Args:
        sweeps: List of sweep strings like ['a=1,2', 'b=x,y']

    Returns:
        List of override strings like ['a=1 b=x', 'a=1 b=y', 'a=2 b=x', 'a=2 b=y']
    """
    if not sweeps:
        return [""]

    parsed = [parse_sweep_param(s) for s in sweeps]
    keys = [p[0] for p in parsed]
    value_lists = [p[1] for p in parsed]

    combinations = []
    for values in itertools.product(*value_lists):
        override = " ".join(f"{k}={v}" for k, v in zip(keys, values, strict=True))
        combinations.append(override)

    return combinations


def run_experiment(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    script: str,
    config_name: str,
    overrides: str,
    gpu_id: int | None,
    job_num: int,
    dry_run: bool = False,
) -> dict:
    """Run a single experiment on a specific GPU or CPU.

    Args:
        script: Path to the Python script to run.
        config_name: Hydra config name.
        overrides: Space-separated Hydra overrides.
        gpu_id: GPU ID to assign via CUDA_VISIBLE_DEVICES, or None for CPU-only.
        job_num: Job number for logging.
        dry_run: If True, print command without executing.

    Returns:
        Dict with job results including status, stdout, stderr.
    """
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

    cmd = [
        "uv",
        "run",
        "python",
        script,
        f"--config-name={config_name}",
    ]

    if overrides:
        cmd.extend(overrides.split())

    device_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    print(f"[Job {job_num}] {device_str}: {' '.join(cmd)}")

    if dry_run:
        return {"job_num": job_num, "gpu": gpu_id, "status": "dry_run", "overrides": overrides}

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        status = "success" if result.returncode == 0 else "failed"
        return {
            "job_num": job_num,
            "gpu": gpu_id,
            "status": status,
            "returncode": result.returncode,
            "overrides": overrides,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return {
            "job_num": job_num,
            "gpu": gpu_id,
            "status": "error",
            "error": str(e),
            "overrides": overrides,
        }


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run multiple Hydra experiments in parallel across GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "script",
        help="Path to the run script (e.g., experiments/training/run.py)",
    )
    parser.add_argument(
        "--config-name",
        "-c",
        required=True,
        help="Hydra config name (e.g., train_small)",
    )
    parser.add_argument(
        "--gpus",
        "-g",
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU only (disables GPU)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (required with --cpu, optional otherwise)",
    )
    parser.add_argument(
        "--sweep",
        "-s",
        action="append",
        default=[],
        help="Sweep parameter (e.g., 'seed=1,2,3'). Can specify multiple times for cartesian product.",
    )
    parser.add_argument(
        "--sweep-file",
        "-f",
        default=None,
        help="Path to YAML file containing sweep parameters.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        default=[],
        help="Explicit override strings to run (alternative to --sweep)",
    )
    parser.add_argument(
        "--max-parallel",
        "-p",
        type=int,
        default=None,
        help="Max parallel jobs (default: number of GPUs)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Determine devices (GPUs or CPU workers)
    gpus: list[int] | None = None
    n_workers: int = 0
    device_desc: str = ""

    if args.cpu:
        if args.workers is None:
            parser.error("--workers is required when using --cpu")
        gpus = None
        n_workers = args.workers
        device_desc = f"{n_workers} CPU workers"
    elif args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        n_workers = args.workers or len(gpus)
        device_desc = f"GPUs {gpus}"
    else:
        parser.error(
            "You must specify devices to use.\n\n"
            "Options:\n"
            "  --gpus 0,1,2,3    Run on specific GPUs (comma-separated IDs)\n"
            "  --cpu --workers 4 Run on CPU only with N parallel workers\n\n"
            "Examples:\n"
            "  simplexity-multirun run.py -c config --gpus 0,1 --sweep 'seed=1,2,3,4'\n"
            "  simplexity-multirun run.py -c config --cpu --workers 4 --sweep 'seed=1,2,3,4'"
        )

    # Collect sweep parameters from --sweep and --sweep-file
    all_sweeps = list(args.sweep)  # Copy to avoid modifying args
    if args.sweep_file:
        all_sweeps.extend(load_sweep_file(args.sweep_file))

    # Generate list of override strings
    if args.overrides:
        override_list = args.overrides
    elif all_sweeps:
        override_list = generate_override_combinations(all_sweeps)
    else:
        override_list = [""]  # Single run with no overrides

    n_jobs = len(override_list)
    max_parallel = args.max_parallel or n_workers

    print(f"Running {n_jobs} experiments across {device_desc}")
    print(f"Max parallel: {max_parallel}")
    print()

    # Assign devices round-robin (GPU IDs or None for CPU)
    jobs = []
    for i, overrides in enumerate(override_list):
        if gpus is not None:
            gpu_id = gpus[i % len(gpus)]
        else:
            gpu_id = None  # CPU mode
        jobs.append((args.script, args.config_name, overrides, gpu_id, i))

    # Run in parallel with staggered starts
    results = []
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}

        # Submit jobs with delay between starts to avoid race conditions
        for i, job in enumerate(jobs):
            if i > 0 and not args.dry_run:
                time.sleep(JOB_START_DELAY_SECONDS)

            future = executor.submit(
                run_experiment,
                script=job[0],
                config_name=job[1],
                overrides=job[2],
                gpu_id=job[3],
                job_num=job[4],
                dry_run=args.dry_run,
            )
            futures[future] = job[4]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status_symbol = "\u2713" if result["status"] == "success" else "\u2717"
            device_str = f"GPU {result['gpu']}" if result["gpu"] is not None else "CPU"
            print(f"[Job {result['job_num']}] {status_symbol} {device_str}: {result['status']}")

            if result["status"] == "failed" and not args.dry_run:
                print(f"  stderr: {result.get('stderr', '')[:500]}")

    # Summary
    print()
    print("=" * 60)
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "failed")
    print(f"Complete: {successes} succeeded, {failures} failed out of {n_jobs} jobs")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
