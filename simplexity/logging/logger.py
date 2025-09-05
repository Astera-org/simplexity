import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class Logger(ABC):
    """Logs to a variety of backends."""

    @abstractmethod
    def log_config(self, config: DictConfig, resolve: bool = False) -> None:
        """Log config to the logger."""
        ...

    @abstractmethod
    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the logger."""
        ...

    @abstractmethod
    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the logger."""
        ...

    @abstractmethod
    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the logger."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        ...

    def _get_git_info(self, repo_path: Path) -> dict[str, str]:
        """Get git repository information.

        Args:
            repo_path: Path to the git repository

        Returns:
            Dictionary with git information (commit, branch, dirty state, remote)
        """
        try:
            # Create a partial function with common arguments
            run = partial(subprocess.run, capture_output=True, text=True, timeout=2.0)
            
            # Get commit hash
            result = run(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
            commit_full = result.stdout.strip() if result.returncode == 0 else "unknown"
            commit_short = commit_full[:8] if commit_full != "unknown" else "unknown"

            # Check if working directory is dirty (has uncommitted changes)
            result = run(["git", "-C", str(repo_path), "diff-index", "--quiet", "HEAD", "--"])
            is_dirty = result.returncode != 0

            # Get current branch name
            result = run(["git", "-C", str(repo_path), "branch", "--show-current"])
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Get remote URL
            result = run(["git", "-C", str(repo_path), "remote", "get-url", "origin"])
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

    def log_git_info(self) -> None:
        """Log git information for reproducibility.

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
