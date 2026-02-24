"""Tests for the run_parallel CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simplexity.cli.run_parallel import (
    Job,
    _run_single_job,
    dispatch_jobs,
    generate_jobs,
    generate_override_combinations,
    load_sweep_file,
    main,
    parse_sweep_param,
)


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


class TestParseSweepParam:
    """Tests for parse_sweep_param."""

    def test_single_value(self) -> None:
        """Verify parsing a single-value sweep string."""
        key, values = parse_sweep_param("seed=42")
        assert key == "seed"
        assert values == ["42"]

    def test_multiple_values(self) -> None:
        """Verify parsing a multi-value sweep string."""
        key, values = parse_sweep_param("lr=0.01,0.001,0.0001")
        assert key == "lr"
        assert values == ["0.01", "0.001", "0.0001"]

    def test_values_with_spaces_are_stripped(self) -> None:
        """Verify whitespace around values is stripped."""
        key, values = parse_sweep_param("a=1, 2, 3")
        assert key == "a"
        assert values == ["1", "2", "3"]

    def test_dotted_key(self) -> None:
        """Verify dotted parameter keys are preserved."""
        key, values = parse_sweep_param("model.n_heads=1,2,4")
        assert key == "model.n_heads"
        assert values == ["1", "2", "4"]

    def test_value_containing_equals(self) -> None:
        """Verify only the first '=' is used as the key-value separator."""
        key, values = parse_sweep_param("path=/a=b,/c=d")
        assert key == "path"
        assert values == ["/a=b", "/c=d"]


class TestGenerateOverrideCombinations:
    """Tests for generate_override_combinations."""

    def test_empty_sweeps(self) -> None:
        """Verify empty input returns a single empty override string."""
        assert generate_override_combinations([]) == [""]

    def test_single_sweep(self) -> None:
        """Verify a single sweep expands to individual overrides."""
        result = generate_override_combinations(["seed=1,2,3"])
        assert result == ["seed=1", "seed=2", "seed=3"]

    def test_two_sweeps_cartesian(self) -> None:
        """Verify two sweeps produce their cartesian product."""
        result = generate_override_combinations(["a=1,2", "b=x,y"])
        assert result == ["a=1 b=x", "a=1 b=y", "a=2 b=x", "a=2 b=y"]

    def test_three_sweeps_cartesian(self) -> None:
        """Verify three sweeps produce their cartesian product."""
        result = generate_override_combinations(["a=1,2", "b=x,y", "c=p,q"])
        assert len(result) == 8
        assert result[0] == "a=1 b=x c=p"
        assert result[-1] == "a=2 b=y c=q"


class TestLoadSweepFile:
    """Tests for load_sweep_file."""

    def test_load_list_values(self, tmp_path: Path) -> None:
        """Verify YAML list values are joined with commas."""
        sweep_file = tmp_path / "sweep.yaml"
        sweep_file.write_text("seed: [1, 2, 3]\nmodel.lr: [0.01, 0.001]\n")
        result = load_sweep_file(str(sweep_file))
        assert result == ["seed=1,2,3", "model.lr=0.01,0.001"]

    def test_load_scalar_value(self, tmp_path: Path) -> None:
        """Verify YAML scalar values are stringified directly."""
        sweep_file = tmp_path / "sweep.yaml"
        sweep_file.write_text("seed: 42\n")
        result = load_sweep_file(str(sweep_file))
        assert result == ["seed=42"]

    def test_load_mixed_values(self, tmp_path: Path) -> None:
        """Verify mixed list and scalar values are handled correctly."""
        sweep_file = tmp_path / "sweep.yaml"
        sweep_file.write_text("seed: [1, 2]\nbatch_size: 64\n")
        result = load_sweep_file(str(sweep_file))
        assert result == ["seed=1,2", "batch_size=64"]


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


class TestRunSingleJob:
    """Tests for _run_single_job."""

    def test_successful_job_returns_success(self) -> None:
        """Verify a zero-returncode subprocess produces a success result."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""

        job = Job(script="train.py", config_name="config", overrides="seed=1", gpu_id=0, job_num=0)

        with patch("simplexity.cli.run_parallel.subprocess.run", return_value=mock_result) as mock_run:
            result = _run_single_job(job)

        assert result["status"] == "success"
        assert result["job_num"] == 0
        assert result["gpu"] == 0
        assert result["returncode"] == 0
        assert result["overrides"] == "seed=1"
        call_env = mock_run.call_args.kwargs["env"]
        assert call_env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_failed_job_returns_failed(self) -> None:
        """Verify a nonzero-returncode subprocess produces a failed result."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error message"

        job = Job(script="train.py", config_name="config", overrides="", gpu_id=1, job_num=3)

        with patch("simplexity.cli.run_parallel.subprocess.run", return_value=mock_result):
            result = _run_single_job(job)

        assert result["status"] == "failed"
        assert result["returncode"] == 1
        assert result["stderr"] == "error message"

    def test_cpu_mode_sets_empty_cuda_visible(self) -> None:
        """Verify CPU-only jobs set CUDA_VISIBLE_DEVICES to empty string."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        job = Job(script="train.py", config_name="config", overrides="", gpu_id=None, job_num=0)

        with patch("simplexity.cli.run_parallel.subprocess.run", return_value=mock_result) as mock_run:
            _run_single_job(job)

        call_env = mock_run.call_args.kwargs["env"]
        assert call_env["CUDA_VISIBLE_DEVICES"] == ""

    def test_exception_returns_error(self) -> None:
        """Verify subprocess exceptions are caught and returned as error status."""
        job = Job(script="train.py", config_name="config", overrides="", gpu_id=0, job_num=0)

        with patch("simplexity.cli.run_parallel.subprocess.run", side_effect=OSError("spawn failed")):
            result = _run_single_job(job)

        assert result["status"] == "error"
        assert "spawn failed" in result["error"]

    def test_long_stdout_truncated(self) -> None:
        """Verify stdout longer than 2000 chars is truncated."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "x" * 5000
        mock_result.stderr = ""

        job = Job(script="train.py", config_name="config", overrides="", gpu_id=0, job_num=0)

        with patch("simplexity.cli.run_parallel.subprocess.run", return_value=mock_result):
            result = _run_single_job(job)

        assert len(result["stdout"]) == 2000


class TestDispatchJobs:
    """Tests for dispatch_jobs."""

    @staticmethod
    def _make_mock_executor(mock_result: dict):
        """Create a mock ProcessPoolExecutor that returns mock_result for every submit."""
        mock_future = MagicMock()
        mock_future.result.return_value = mock_result

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future
        return mock_executor, mock_future

    @staticmethod
    def _success_result(**overrides: object) -> dict:
        base = {
            "job_num": 0,
            "gpu": 0,
            "status": "success",
            "returncode": 0,
            "overrides": "",
            "stdout": "",
            "stderr": "",
        }
        base.update(overrides)
        return base

    @staticmethod
    def _failed_result(**overrides: object) -> dict:
        base = {
            "job_num": 0,
            "gpu": 0,
            "status": "failed",
            "returncode": 1,
            "overrides": "",
            "stdout": "",
            "stderr": "",
        }
        base.update(overrides)
        return base

    def test_dispatches_all_jobs(self) -> None:
        """Verify all submitted jobs produce results."""
        jobs = [
            Job(script="train.py", config_name="config", overrides=f"seed={i}", gpu_id=0, job_num=i) for i in range(3)
        ]

        mock_result = self._success_result()
        mock_executor, mock_future = self._make_mock_executor(mock_result)

        with (
            patch("simplexity.cli.run_parallel.ProcessPoolExecutor", return_value=mock_executor),
            patch("simplexity.cli.run_parallel.as_completed", return_value=[mock_future] * 3),
            patch("simplexity.cli.run_parallel.time.sleep"),
        ):
            results = dispatch_jobs(jobs, max_parallel=2)

        assert len(results) == 3

    def test_reports_failed_jobs(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify failed job stderr is printed to stdout."""
        jobs = [Job(script="train.py", config_name="config", overrides="", gpu_id=0, job_num=0)]

        mock_result = self._failed_result(stderr="some error")
        mock_executor, mock_future = self._make_mock_executor(mock_result)

        with (
            patch("simplexity.cli.run_parallel.ProcessPoolExecutor", return_value=mock_executor),
            patch("simplexity.cli.run_parallel.as_completed", return_value=[mock_future]),
            patch("simplexity.cli.run_parallel.time.sleep"),
        ):
            results = dispatch_jobs(jobs, max_parallel=1)

        assert results[0]["status"] == "failed"
        captured = capsys.readouterr()
        assert "some error" in captured.out

    def test_cpu_mode_display(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify CPU mode jobs display 'CPU' in output."""
        jobs = [Job(script="train.py", config_name="config", overrides="", gpu_id=None, job_num=0)]

        mock_result = self._success_result(gpu=None)
        mock_executor, mock_future = self._make_mock_executor(mock_result)

        with (
            patch("simplexity.cli.run_parallel.ProcessPoolExecutor", return_value=mock_executor),
            patch("simplexity.cli.run_parallel.as_completed", return_value=[mock_future]),
            patch("simplexity.cli.run_parallel.time.sleep"),
        ):
            results = dispatch_jobs(jobs, max_parallel=1)

        assert results[0]["status"] == "success"
        captured = capsys.readouterr()
        assert "CPU" in captured.out


class TestMain:
    """Tests for the main CLI entry point."""

    @staticmethod
    def _argv(*args: str) -> list[str]:
        return ["prog", "train.py", "-c", "config", *args]

    def test_dry_run_prints_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --dry-run prints job commands without executing."""
        argv = self._argv("--gpus", "0,1", "--sweep", "seed=1,2", "--dry-run")
        with patch("sys.argv", argv):
            main()

        captured = capsys.readouterr()
        assert "[Job 0]" in captured.out
        assert "[Job 1]" in captured.out
        assert "GPU 0" in captured.out
        assert "GPU 1" in captured.out

    def test_cpu_dry_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --cpu --dry-run shows CPU worker count."""
        argv = self._argv("--cpu", "--workers", "2", "--sweep", "seed=1,2", "--dry-run")
        with patch("sys.argv", argv):
            main()

        captured = capsys.readouterr()
        assert "CPU" in captured.out
        assert "2 CPU workers" in captured.out

    def test_no_device_exits_with_error(self) -> None:
        """Verify missing --gpus/--cpu exits with an error."""
        argv = self._argv("--sweep", "seed=1")
        with patch("sys.argv", argv), pytest.raises(SystemExit):
            main()

    def test_cpu_without_workers_exits_with_error(self) -> None:
        """Verify --cpu without --workers exits with an error."""
        argv = self._argv("--cpu", "--sweep", "seed=1")
        with patch("sys.argv", argv), pytest.raises(SystemExit):
            main()

    def test_sweep_file_integration(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --sweep-file loads parameters and generates jobs."""
        sweep_file = tmp_path / "sweep.yaml"
        sweep_file.write_text("seed: [1, 2]\n")

        argv = self._argv("--gpus", "0", "--sweep-file", str(sweep_file), "--dry-run")
        with patch("sys.argv", argv):
            main()

        captured = capsys.readouterr()
        assert "[Job 0]" in captured.out
        assert "[Job 1]" in captured.out

    def test_successful_run_exits_cleanly(self) -> None:
        """Verify a fully successful run does not call sys.exit."""
        mock_result = {
            "job_num": 0,
            "gpu": 0,
            "status": "success",
            "returncode": 0,
            "overrides": "seed=1",
            "stdout": "",
            "stderr": "",
        }
        argv = self._argv("--gpus", "0", "--sweep", "seed=1")

        with (
            patch("sys.argv", argv),
            patch("simplexity.cli.run_parallel.dispatch_jobs", return_value=[mock_result]),
        ):
            main()

    def test_failed_run_exits_with_code_1(self) -> None:
        """Verify any failed job causes sys.exit(1)."""
        mock_result = {
            "job_num": 0,
            "gpu": 0,
            "status": "failed",
            "returncode": 1,
            "overrides": "seed=1",
            "stdout": "",
            "stderr": "err",
        }
        argv = self._argv("--gpus", "0", "--sweep", "seed=1")

        with (
            patch("sys.argv", argv),
            patch("simplexity.cli.run_parallel.dispatch_jobs", return_value=[mock_result]),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_max_parallel_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --max-parallel is displayed in output."""
        argv = self._argv("--gpus", "0,1", "--max-parallel", "1", "--sweep", "seed=1,2", "--dry-run")
        with patch("sys.argv", argv):
            main()

        captured = capsys.readouterr()
        assert "Max parallel: 1" in captured.out
