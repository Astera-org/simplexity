from pathlib import Path


def get_checkpoint_path(directory: Path, step: int, filename: str = "model.pt") -> Path:
    """Construct checkpoint path following the standard naming convention.

    Args:
        directory: Base directory for checkpoints
        step: Training step number
        filename: Checkpoint filename (default: "model.pt")

    Returns:
        Path to checkpoint file: {directory}/{step}/{filename}

    Examples:
        >>> get_checkpoint_path(Path("checkpoints"), 12345)
        PosixPath('checkpoints/12345/model.pt')
        >>> get_checkpoint_path(Path("weights"), 100, "state.pt")
        PosixPath('weights/100/state.pt')
    """
    return directory / str(step) / filename


def parse_checkpoint_step(path: str) -> int | None:
    """Extract training step number from checkpoint path.

    Handles the format: {step}/model.pt or {step}/{filename}

    Args:
        path: File path or S3 key containing checkpoint

    Returns:
        Step number if found, None otherwise

    Examples:
        >>> parse_checkpoint_step("checkpoints/12345/model.pt")
        12345
        >>> parse_checkpoint_step("12345/model.pt")
        12345
    """
    parts = path.split("/")
    if len(parts) >= 2 and parts[-1].endswith(".pt"):
        try:
            return int(parts[-2])
        except ValueError:
            pass

    return None


def format_step_number(step: int, max_steps: int) -> str:
    """Format step number with appropriate zero-padding.

    Args:
        step: Current training step
        max_steps: Maximum number of training steps

    Returns:
        Zero-padded step string

    Examples:
        >>> format_step_number(42, max_steps=100000)
        '000042'
        >>> format_step_number(999, max_steps=999)
        '999'
    """
    assert 0 <= step <= max_steps, f"Step {step} must be between 0 and {max_steps}"
    width = len(str(max_steps))
    return f"{step:0{width}d}"
