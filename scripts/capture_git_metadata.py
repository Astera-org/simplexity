"""Build script to capture git metadata at build time."""

import json
import subprocess
from pathlib import Path


def capture_git_metadata() -> dict[str, str]:
    """Capture git metadata from the current repository.

    Returns:
        Dictionary containing git metadata (commit, branch, dirty state, remote).
        Returns empty dict if git is not available or not in a git repo.
    """
    try:

        def run_git(cmd: list[str]) -> str:
            """Run a git command and return the output."""
            result = subprocess.run(
                ["git"] + cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get commit hash
        commit_full = run_git(["rev-parse", "HEAD"])
        commit_short = commit_full[:8] if commit_full != "unknown" else "unknown"

        # Check if working directory is dirty (has uncommitted changes)
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        is_dirty = result.returncode != 0

        # Get current branch name
        branch = run_git(["branch", "--show-current"])

        # Get remote URL (sanitized version will be handled in the logger)
        remote_url = run_git(["remote", "get-url", "origin"])

        return {
            "commit": commit_short,
            "commit_full": commit_full,
            "dirty": str(is_dirty),
            "branch": branch,
            "remote": remote_url,
        }
    except Exception:
        # Return empty dict if git is not available or repo is not a git repo
        return {}


def main() -> None:
    """Main function to capture and save git metadata."""
    # Capture git metadata
    git_metadata = capture_git_metadata()

    if git_metadata:
        # Create the metadata file in the simplexity package directory
        package_dir = Path("simplexity")
        metadata_file = package_dir / "_git_metadata.json"

        # Ensure the package directory exists
        package_dir.mkdir(exist_ok=True)

        # Write the git metadata to the file
        with open(metadata_file, "w") as f:
            json.dump(git_metadata, f, indent=2)

        print(f"Git metadata captured and saved to {metadata_file}")
    else:
        print("No git metadata available to capture")


if __name__ == "__main__":
    main()
