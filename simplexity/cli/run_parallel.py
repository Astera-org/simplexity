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
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

JOB_START_DELAY_SECONDS = 5


@dataclass(frozen=True)
class Job:
    """Represents a single experiment job to be executed."""

    script: str
    config_name: str
    overrides: str
    gpu_id: int | None
    job_num: int

    def to_cmd(self) -> list[str]:
        """Render the full command list for this job."""
        cmd = [
            "uv",
            "run",
            "python",
            self.script,
            f"--config-name={self.config_name}",
        ]

        if self.overrides:
            cmd.extend(self.overrides.split())

        return cmd

    @property
    def device_str(self) -> str:
        """Human-readable device description."""
        return f"GPU {self.gpu_id}" if self.gpu_id is not None else "CPU"


def load_sweep_file(path: str) -> list[str]:
    """Load sweep parameters from a YAML file."""
    cfg = OmegaConf.load(path)
    sweeps = []
    for key, values in cfg.items():
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
    """Generate all combinations of sweep parameters (cartesian product)."""
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


def generate_jobs(
    script: str,
    config_name: str,
    sweeps: list[str],
    overrides: list[str],
    gpus: list[int] | None,
) -> list[Job]:
    """Generate a list of jobs from sweep parameters and device configuration."""
    if overrides:
        override_list = overrides
    elif sweeps:
        override_list = generate_override_combinations(sweeps)
    else:
        override_list = [""]

    jobs = []
    for i, override_str in enumerate(override_list):
        gpu_id = gpus[i % len(gpus)] if gpus is not None else None
        jobs.append(
            Job(
                script=script,
                config_name=config_name,
                overrides=override_str,
                gpu_id=gpu_id,
                job_num=i,
            )
        )

    return jobs


def _run_single_job(job: Job) -> dict:
    """Run a single experiment job in a subprocess."""
    env = os.environ.copy()
    if job.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = job.to_cmd()

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
            "job_num": job.job_num,
            "gpu": job.gpu_id,
            "status": status,
            "returncode": result.returncode,
            "overrides": job.overrides,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return {
            "job_num": job.job_num,
            "gpu": job.gpu_id,
            "status": "error",
            "error": str(e),
            "overrides": job.overrides,
        }


def dispatch_jobs(jobs: list[Job], max_parallel: int) -> list[dict]:
    """Execute jobs in parallel with staggered starts."""
    results = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}

        for i, job in enumerate(jobs):
            if i > 0:
                time.sleep(JOB_START_DELAY_SECONDS)

            print(f"[Job {job.job_num}] {job.device_str}: {' '.join(job.to_cmd())}")
            future = executor.submit(_run_single_job, job)
            futures[future] = job.job_num

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status_symbol = "\u2713" if result["status"] == "success" else "\u2717"
            device_str = f"GPU {result['gpu']}" if result["gpu"] is not None else "CPU"
            print(f"[Job {result['job_num']}] {status_symbol} {device_str}: {result['status']}")

            if result["status"] == "failed":
                print(f"  stderr: {result.get('stderr', '')[:500]}")

    return results


def main() -> None:
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

    all_sweeps = list(args.sweep)
    if args.sweep_file:
        all_sweeps.extend(load_sweep_file(args.sweep_file))

    jobs = generate_jobs(
        script=args.script,
        config_name=args.config_name,
        sweeps=all_sweeps,
        overrides=args.overrides,
        gpus=gpus,
    )

    n_jobs = len(jobs)
    max_parallel = args.max_parallel or n_workers

    print(f"Running {n_jobs} experiments across {device_desc}")
    print(f"Max parallel: {max_parallel}")
    print()

    if args.dry_run:
        for job in jobs:
            print(f"[Job {job.job_num}] {job.device_str}: {' '.join(job.to_cmd())}")
        return

    results = dispatch_jobs(jobs, max_parallel)

    print()
    print("=" * 60)
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "failed")
    print(f"Complete: {successes} succeeded, {failures} failed out of {n_jobs} jobs")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
