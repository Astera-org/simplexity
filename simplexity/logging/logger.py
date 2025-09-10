import json
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import matplotlib.figure
import mlflow
import numpy
import PIL.Image
import plotly.graph_objects
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
    def log_figure(
        self,
        figure: matplotlib.figure.Figure | plotly.graph_objects.Figure,
        artifact_file: str,
        **kwargs,
    ) -> None:
        """Log a figure to the logger."""
        ...

    @abstractmethod
    def log_image(
        self,
        image: numpy.ndarray | PIL.Image.Image | mlflow.Image,
        artifact_file: str | None = None,
        key: str | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        """Log an image to the logger.

        Args:
            image: Image to log (numpy array, PIL Image, or mlflow Image)
            artifact_file: File path for artifact mode (e.g., "image.png")
            key: Key name for time-stepped mode (requires step parameter)
            step: Step number for time-stepped mode (requires key parameter)
            **kwargs: Additional arguments passed to the underlying save method

        Note:
            Must provide either artifact_file OR both key and step parameters.
            Providing neither or only one of key/step will result in an error.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        ...

    def _sanitize_remote(self, remote: str) -> str:
        """Sanitize git remote URL to remove potential credentials.

        Args:
            remote: The git remote URL

        Returns:
            Sanitized remote URL without credentials
        """
        if not remote:
            return remote

        # Try URL-like first: http(s)://..., ssh://..., git+https://...
        try:
            parts = urlsplit(remote)
            if parts.scheme:
                # rebuild without username/password, query, fragment
                host = parts.hostname or ""
                port = f":{parts.port}" if parts.port else ""
                path = parts.path or ""
                return f"{parts.scheme}://{host}{port}{path}"
        except Exception:
            pass

        # SCP-like: user@host:path
        m = re.match(r"^[^@]+@([^:]+):(.*)$", remote)
        if m:
            host, path = m.groups()
            return f"{host}:{path}"

        # Otherwise return as-is
        return remote

    def _find_git_root(self, start: Path) -> Path | None:
        """Find the git repository root from a starting path.

        Args:
            start: Starting path to search from

        Returns:
            Path to git repository root, or None if not found
        """
        try:
            r = subprocess.run(
                ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if r.returncode == 0:
                return Path(r.stdout.strip())
        except Exception:
            pass
        for parent in [start.resolve(), *start.resolve().parents]:
            if (parent / ".git").exists():
                return parent
        return None

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

            # Get remote URL and sanitize it
            result = run(["git", "-C", str(repo_path), "remote", "get-url", "origin"])
            remote_url = result.stdout.strip() if result.returncode == 0 else "unknown"
            remote_url = self._sanitize_remote(remote_url)

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

    def _load_git_metadata(self, pkg_dir: Path) -> dict[str, str]:
        """Load git metadata from build-time capture file.

        Args:
            pkg_dir: Path to the package directory

        Returns:
            Dictionary with git metadata, or empty dict if not available
        """
        try:
            metadata_file = pkg_dir / "_git_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                # Sanitize the remote URL if it exists
                if "remote" in metadata:
                    metadata["remote"] = self._sanitize_remote(metadata["remote"])
                return metadata
        except Exception:
            pass
        return {}

    def _is_installed_package(self, pkg_dir: Path) -> bool:
        """Check if the package is installed (in site-packages) vs editable install.

        Args:
            pkg_dir: Path to the package directory

        Returns:
            True if installed in site-packages, False if editable install
        """
        # Check if we're in site-packages
        pkg_path_str = str(pkg_dir.resolve())
        return "site-packages" in pkg_path_str

    def log_git_info(self) -> None:
        """Log git information for reproducibility.

        Logs git information for both the main repository (where the training
        script is running) and the simplexity library repository. For simplexity,
        uses live git commands if in development (editable install), otherwise
        uses build-time captured metadata for installed packages.
        """
        tags = {}

        # Track main repository (current working directory)
        main_root = self._find_git_root(Path.cwd())
        if main_root:
            for k, v in self._get_git_info(main_root).items():
                tags[f"git.main.{k}"] = v

        # Track simplexity repository with improved logic for installed packages
        pkg_dir = Path(__file__).resolve().parent

        if self._is_installed_package(pkg_dir):
            # Package is installed in site-packages - try to load build-time metadata
            git_info = self._load_git_metadata(pkg_dir)
            if git_info:
                for k, v in git_info.items():
                    tags[f"git.simplexity.{k}"] = v
            # If no build-time metadata available, skip simplexity tracking
        else:
            # Package is in development mode (editable install) - use live git
            simplexity_root = self._find_git_root(pkg_dir)
            if simplexity_root:
                for k, v in self._get_git_info(simplexity_root).items():
                    tags[f"git.simplexity.{k}"] = v

        if tags:
            self.log_tags(tags)
