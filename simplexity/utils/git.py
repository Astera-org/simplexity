import re
import subprocess
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit


def _sanitize_remote(remote: str) -> str:
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


def _find_git_root(start: Path) -> Path | None:
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


def get_git_info(repo_path: Path | None = None) -> dict[str, str]:
    """Get git repository information.

    Args:
        repo_path: Path to the git repository

    Returns:
        Dictionary with git information (commit, branch, dirty state, remote)
    """
    if repo_path is None:
        repo_path = _find_git_root(Path.cwd())
    if repo_path is None:
        return {}
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
        remote_url = _sanitize_remote(remote_url)

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
