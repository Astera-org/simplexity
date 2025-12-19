"""Tests for the run_parallel CLI module."""

import pytest

from simplexity.cli.run_parallel import Job, generate_jobs


class TestJob:
    """Tests for the Job dataclass."""

    def test_to_cmd_without_overrides(self) -> None:
        """Verify to_cmd() produces correct command without overrides."""
        job = Job(
            script="train.py",
            config_name="config",
            overrides="",
            gpu_id=0,
            job_num=0,
        )
        assert job.to_cmd() == ["uv", "run", "python", "train.py", "--config-name=config"]

    def test_to_cmd_with_overrides(self) -> None:
        """Verify to_cmd() appends overrides to the command."""
        job = Job(
            script="train.py",
            config_name="config",
            overrides="seed=42 lr=0.01",
            gpu_id=0,
            job_num=0,
        )
        assert job.to_cmd() == [
            "uv",
            "run",
            "python",
            "train.py",
            "--config-name=config",
            "seed=42",
            "lr=0.01",
        ]

    def test_device_str_gpu(self) -> None:
        """Verify device_str shows GPU ID when gpu_id is set."""
        job = Job(script="train.py", config_name="config", overrides="", gpu_id=2, job_num=0)
        assert job.device_str == "GPU 2"

    def test_device_str_cpu(self) -> None:
        """Verify device_str shows CPU when gpu_id is None."""
        job = Job(script="train.py", config_name="config", overrides="", gpu_id=None, job_num=0)
        assert job.device_str == "CPU"


class TestGenerateJobs:
    """Tests for the generate_jobs function."""

    def test_gpu_round_robin_assignment(self) -> None:
        """Verify GPUs are assigned round-robin across jobs."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3,4,5,6"],
            overrides=[],
            gpus=[0, 1],
        )

        assert len(jobs) == 6
        assert [job.gpu_id for job in jobs] == [0, 1, 0, 1, 0, 1]

    def test_gpu_round_robin_with_three_gpus(self) -> None:
        """Verify round-robin with 3 GPUs and 5 jobs."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3,4,5"],
            overrides=[],
            gpus=[0, 2, 4],
        )

        assert len(jobs) == 5
        assert [job.gpu_id for job in jobs] == [0, 2, 4, 0, 2]

    def test_cpu_mode_assigns_none(self) -> None:
        """Verify CPU mode assigns None for all gpu_ids."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3"],
            overrides=[],
            gpus=None,
        )

        assert len(jobs) == 3
        assert all(job.gpu_id is None for job in jobs)

    def test_sweep_cartesian_product(self) -> None:
        """Verify sweeps produce cartesian product of overrides."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["a=1,2", "b=x,y"],
            overrides=[],
            gpus=[0],
        )

        assert len(jobs) == 4
        overrides = [job.overrides for job in jobs]
        assert overrides == ["a=1 b=x", "a=1 b=y", "a=2 b=x", "a=2 b=y"]

    def test_sweep_single_param(self) -> None:
        """Verify single sweep parameter generates correct jobs."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3"],
            overrides=[],
            gpus=[0, 1],
        )

        assert len(jobs) == 3
        assert [job.overrides for job in jobs] == ["seed=1", "seed=2", "seed=3"]

    def test_explicit_overrides_used_instead_of_sweeps(self) -> None:
        """Verify explicit overrides take precedence over sweeps."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3"],
            overrides=["custom=a", "custom=b"],
            gpus=[0],
        )

        assert len(jobs) == 2
        assert [job.overrides for job in jobs] == ["custom=a", "custom=b"]

    def test_no_sweeps_or_overrides_creates_single_job(self) -> None:
        """Verify empty sweeps and overrides creates one job with empty overrides."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=[],
            overrides=[],
            gpus=[0],
        )

        assert len(jobs) == 1
        assert jobs[0].overrides == ""

    def test_job_numbers_sequential(self) -> None:
        """Verify job numbers are assigned sequentially starting from 0."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=["seed=1,2,3,4"],
            overrides=[],
            gpus=[0, 1],
        )

        assert [job.job_num for job in jobs] == [0, 1, 2, 3]

    def test_script_and_config_propagated(self) -> None:
        """Verify script and config_name are correctly set on all jobs."""
        jobs = generate_jobs(
            script="experiments/run.py",
            config_name="my_config",
            sweeps=["seed=1,2"],
            overrides=[],
            gpus=[0],
        )

        assert all(job.script == "experiments/run.py" for job in jobs)
        assert all(job.config_name == "my_config" for job in jobs)

    @pytest.mark.parametrize(
        ("sweeps", "expected_count"),
        [
            (["a=1,2,3"], 3),
            (["a=1,2", "b=1,2"], 4),
            (["a=1,2", "b=1,2,3"], 6),
            (["a=1,2", "b=1,2", "c=1,2"], 8),
        ],
    )
    def test_cartesian_product_counts(self, sweeps: list[str], expected_count: int) -> None:
        """Verify correct number of jobs for various cartesian product sizes."""
        jobs = generate_jobs(
            script="train.py",
            config_name="config",
            sweeps=sweeps,
            overrides=[],
            gpus=[0],
        )

        assert len(jobs) == expected_count
