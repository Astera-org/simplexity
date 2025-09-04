import os
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dotenv
import mlflow
from mlflow.entities import Metric, Param, RunTag
from omegaconf import DictConfig, OmegaConf

from simplexity.logging.logger import Logger

dotenv.load_dotenv()
_DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")


class MLFlowLogger(Logger):
    """Logs to MLflow Tracking."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow logger."""
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._client.create_experiment(experiment_name)
        run = self._client.create_run(experiment_id=experiment_id, run_name=run_name)
        self._run_id = run.info.run_id

        # Automatically log git information for reproducibility
        self._log_git_info()

    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to MLflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            OmegaConf.save(config, config_path, resolve=resolve)
            self._client.log_artifact(self._run_id, config_path)

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to MLflow."""
        timestamp = int(time.time() * 1000)
        metrics = self._flatten_metric_dict(metric_dict, timestamp, step)
        self._log_batch(metrics=metrics)

    def _flatten_metric_dict(
        self, metric_dict: Mapping[str, Any], timestamp: int, step: int, key_prefix: str = ""
    ) -> list[Metric]:
        """Flatten a dictionary of metrics into a list of Metric entities."""
        metrics = []
        for key, value in metric_dict.items():
            key = f"{key_prefix}/{key}" if key_prefix else key
            if isinstance(value, Mapping):
                nested_metrics = self._flatten_metric_dict(value, timestamp, step, key_prefix=key)
                metrics.extend(nested_metrics)
            else:
                value = float(value)
                metric = Metric(key, value, timestamp, step)
                metrics.append(metric)
        return metrics

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to MLflow."""
        params = self._flatten_param_dict(param_dict)
        self._log_batch(params=params)

    def _flatten_param_dict(self, param_dict: Mapping[str, Any], key_prefix: str = "") -> list[Param]:
        """Flatten a dictionary of params into a list of Param entities."""
        params = []
        for key, value in param_dict.items():
            key = f"{key_prefix}.{key}" if key_prefix else key
            if isinstance(value, Mapping):
                nested_params = self._flatten_param_dict(value, key_prefix=key)
                params.extend(nested_params)
            else:
                value = str(value)
                param = Param(key, value)
                params.append(param)
        return params

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Set tags on the MLFlow."""
        tags = [RunTag(k, str(v)) for k, v in tag_dict.items()]
        self._log_batch(tags=tags)

    def close(self):
        """End the MLflow run."""
        self._client.set_terminated(self._run_id)

    def _log_batch(self, **kwargs: Any) -> None:
        """Log arbitrary data to MLflow."""
        self._client.log_batch(self._run_id, **kwargs, synchronous=False)

    def _get_git_info(self, repo_path: Path) -> dict[str, str]:
        """Get git repository information.

        Args:
            repo_path: Path to the git repository

        Returns:
            Dictionary with git information (commit, branch, dirty state, remote)
        """
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, timeout=2
            )
            commit_full = result.stdout.strip() if result.returncode == 0 else "unknown"
            commit_short = commit_full[:8] if commit_full != "unknown" else "unknown"

            # Check if working directory is dirty (has uncommitted changes)
            result = subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True, timeout=2
            )
            is_dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

            # Get current branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path, capture_output=True, text=True, timeout=2
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=2,
            )
            remote_url = result.stdout.strip() if result.returncode == 0 else "unknown"

            return {
                "commit": commit_short,
                "commit_full": commit_full,
                "dirty": str(is_dirty),
                "branch": branch,
                "remote": remote_url,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Return empty dict if git is not available or repo is not a git repo
            return {}

    def _log_git_info(self) -> None:
        """Automatically log git information for reproducibility.

        Logs git information for both the main repository (where the training
        script is running) and the simplexity library repository.
        """
        tags = {}

        # Track main repository (current working directory)
        main_repo_info = self._get_git_info(Path.cwd())
        if main_repo_info:
            for key, value in main_repo_info.items():
                tags[f"git.main.{key}"] = value

        # Track simplexity repository
        try:
            import simplexity

            # Try multiple ways to find simplexity path
            simplexity_path = None

            # Method 1: Use __file__ if available
            file_attr = getattr(simplexity, "__file__", None)
            if file_attr:
                simplexity_path = Path(file_attr).parent.parent
            # Method 2: Use __path__ for namespace packages
            else:
                path_attr = getattr(simplexity, "__path__", None)
                if path_attr:
                    # path_attr might be a _NamespacePath or similar iterable
                    for path in path_attr:
                        if path:
                            simplexity_path = Path(path).parent
                            break
            # Method 3: Use the module spec
            if not simplexity_path:
                import importlib.util

                spec = importlib.util.find_spec("simplexity")
                if spec and spec.origin:
                    simplexity_path = Path(spec.origin).parent.parent

            if simplexity_path and simplexity_path.exists():
                simplexity_info = self._get_git_info(simplexity_path)
                if simplexity_info:
                    for key, value in simplexity_info.items():
                        tags[f"git.simplexity.{key}"] = value
        except (ImportError, AttributeError, TypeError):
            pass

        # Log all git tags if we found any
        if tags:
            self.log_tags(tags)

    def log_storage_info(self, persister: Any) -> None:
        """Log model storage information for tracking.

        Args:
            persister: Model persister object (S3Persister, LocalPersister, etc.)
        """
        tags = {}

        # Check if it's an S3Persister
        if hasattr(persister, "bucket") and hasattr(persister, "prefix"):
            tags["storage.type"] = "s3"
            tags["storage.location"] = f"s3://{persister.bucket}/{persister.prefix}"
            tags["storage.bucket"] = persister.bucket
            tags["storage.prefix"] = persister.prefix
        # Check if it's a LocalPersister or has a directory attribute
        elif hasattr(persister, "directory"):
            tags["storage.type"] = "local"
            tags["storage.location"] = str(Path(persister.directory).absolute())
        else:
            tags["storage.type"] = "unknown"

        if tags:
            self.log_tags(tags)
