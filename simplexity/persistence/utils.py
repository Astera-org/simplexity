import re


def parse_checkpoint_step(path: str) -> int | None:
    """Extract training step number from checkpoint path.

    Handles multiple formats:
    - step_12345.pt / step-12345.pt
    - 12345/model.pt
    - model_weights/step_00012345.pt

    Args:
        path: File path or S3 key containing checkpoint

    Returns:
        Step number if found, None otherwise

    Examples:
        >>> parse_checkpoint_step("model_weights/step_12345.pt")
        12345
        >>> parse_checkpoint_step("checkpoints/12345/model.pt")
        12345
        >>> parse_checkpoint_step("step-00500.pt")
        500
    """
    m = re.search(r"step[_-]?(\d+)\.pt$", path)
    if m:
        return int(m.group(1))

    parts = path.split("/")
    if parts and parts[-1] == "model.pt" and len(parts) >= 2:
        try:
            return int(parts[-2])
        except ValueError:
            pass

    return None


def compute_step_width(max_steps: int) -> int:
    """Compute zero-padding width for step numbers.

    Ensures lexicographic sorting matches chronological order.

    Args:
        max_steps: Maximum number of training steps

    Returns:
        Number of digits to use for zero-padding

    Examples:
        >>> compute_step_width(999)
        3
        >>> compute_step_width(100000)
        6
    """
    return len(str(max_steps))


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
    width = compute_step_width(max_steps)
    return f"{step:0{width}d}"
