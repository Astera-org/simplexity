from pathlib import Path

SUPPORTED_EXTENSIONS = (".pt", ".eqx", ".pkl", ".ckpt", ".pth")


def _is_valid_checkpoint_filename(filename: str) -> bool:
    """Check if filename is a valid checkpoint filename with supported extension.

    Args:
        filename: The checkpoint filename to validate

    Returns:
        True if filename has a supported extension, False otherwise

    Examples:
        >>> _is_valid_checkpoint_filename("model.pt")
        True
        >>> _is_valid_checkpoint_filename("state.eqx")
        True
        >>> _is_valid_checkpoint_filename("invalid.txt")
        False
    """
    return filename.endswith(SUPPORTED_EXTENSIONS)


def get_checkpoint_path(
    directory: Path, step: int, filename: str = "model.pt", max_steps: int | None = None
) -> Path:
    """Construct checkpoint path following the standard naming convention.

    Args:
        directory: Base directory for checkpoints
        step: Training step number (must be non-negative)
        filename: Checkpoint filename (default: "model.pt")
        max_steps: Maximum number of training steps. If provided, step will be zero-padded

    Returns:
        Path to checkpoint file: {directory}/{step}/{filename}

    Raises:
        ValueError: If step is negative or filename has unsupported extension

    Examples:
        >>> get_checkpoint_path(Path("checkpoints"), 12345)
        PosixPath('checkpoints/12345/model.pt')
        >>> get_checkpoint_path(Path("weights"), 100, "state.pt")
        PosixPath('weights/100/state.pt')
        >>> get_checkpoint_path(Path("checkpoints"), 42, max_steps=100000)
        PosixPath('checkpoints/000042/model.pt')
    """
    if step < 0:
        raise ValueError(f"Step must be non-negative, got {step}")
    if not _is_valid_checkpoint_filename(filename):
        raise ValueError(f"Filename must have one of these extensions: {SUPPORTED_EXTENSIONS}, got {filename}")

    if max_steps is not None:
        step_str = format_step_number(step, max_steps)
    else:
        step_str = str(step)

    return directory / step_str / filename


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
    if len(parts) >= 2 and _is_valid_checkpoint_filename(parts[-1]):
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

    Raises:
        ValueError: If step is not between 0 and max_steps

    Examples:
        >>> format_step_number(42, max_steps=100000)
        '000042'
        >>> format_step_number(999, max_steps=999)
        '999'
    """
    if not 0 <= step <= max_steps:
        raise ValueError(f"Step {step} must be between 0 and {max_steps}")
    width = len(str(max_steps))
    return f"{step:0{width}d}"
