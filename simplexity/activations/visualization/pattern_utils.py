"""Shared pattern detection, parsing, and substitution utilities."""

from __future__ import annotations

import re

from simplexity.exceptions import ConfigValidationError

# Compiled regex for range patterns (e.g., "0...10")
RANGE_PATTERN = re.compile(r"(\d+)\.\.\.(\d+)")


def count_patterns(text: str) -> int:
    """Count wildcard (*) and range (N...M) patterns in text.

    Args:
        text: String to check for patterns

    Returns:
        Total number of wildcard and range patterns found
    """
    return text.count("*") + len(RANGE_PATTERN.findall(text))


def has_pattern(text: str) -> bool:
    """Check if text contains any wildcard (*) or range (N...M) pattern.

    Args:
        text: String to check for patterns

    Returns:
        True if text contains at least one pattern
    """
    return "*" in text or bool(RANGE_PATTERN.search(text))


def validate_single_pattern(text: str, context: str) -> None:
    """Validate that text has at most one pattern.

    Args:
        text: String to validate
        context: Description for error message (e.g., "Key", "Field name")

    Raises:
        ConfigValidationError: If text contains multiple patterns
    """
    if count_patterns(text) > 1:
        raise ConfigValidationError(f"{context} cannot have multiple patterns: {text}")


def substitute_pattern(text: str, index: int) -> str:
    """Replace the first wildcard or range pattern with an index.

    Handles both wildcard (*) and range (N...M) patterns. If both are present,
    wildcard takes precedence.

    Args:
        text: String containing a pattern
        index: Index value to substitute

    Returns:
        Text with first pattern replaced by index
    """
    if "*" in text:
        return text.replace("*", str(index), 1)
    return RANGE_PATTERN.sub(str(index), text, count=1)


def substitute_range(text: str, index: int) -> str:
    """Replace a range pattern (N...M) with an index.

    Args:
        text: String containing a range pattern
        index: Index value to substitute

    Returns:
        Text with range pattern replaced by index
    """
    return RANGE_PATTERN.sub(str(index), text, count=1)


def parse_range(text: str) -> tuple[int, int] | None:
    """Extract (start, end) from a range pattern.

    Args:
        text: String potentially containing a range pattern like "0...10"

    Returns:
        Tuple of (start, end) if range found, None otherwise
    """
    match = RANGE_PATTERN.search(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def is_valid_range(text: str) -> bool:
    """Check if text is a valid range pattern with start < end.

    Args:
        text: String to check (e.g., "0...10")

    Returns:
        True if text is a valid range pattern with start < end
    """
    result = parse_range(text)
    if result is None:
        return False
    start, end = result
    return start < end


def build_wildcard_regex(pattern: str, capture: str = r"(\d+)") -> re.Pattern[str]:
    """Build a regex pattern from a wildcard pattern.

    Escapes special regex characters and replaces * with a capture group.

    Args:
        pattern: String with * wildcard (e.g., "factor_*/projected")
        capture: Regex capture group to replace * with (default: numeric capture)

    Returns:
        Compiled regex pattern for matching
    """
    escaped = re.escape(pattern).replace(r"\*", capture)
    return re.compile(f"^{escaped}$")


__all__ = [
    "RANGE_PATTERN",
    "build_wildcard_regex",
    "count_patterns",
    "has_pattern",
    "is_valid_range",
    "parse_range",
    "substitute_pattern",
    "substitute_range",
    "validate_single_pattern",
]
