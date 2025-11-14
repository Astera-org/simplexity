"""Subprocess utilities."""

import subprocess
from functools import wraps
from typing import Any


def handle_subprocess_errors(default_return: Any | None = None):
    """Decorator to catch subprocess errors and return a default value.

    Args:
        default_return: Value to return on subprocess errors. Defaults to empty dict.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return default_return

        return wrapper

    return decorator
