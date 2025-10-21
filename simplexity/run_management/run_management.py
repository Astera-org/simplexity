from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig


def _get_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DictConfig:
    """Get the config from the arguments."""
    if kwargs and "cfg" in kwargs:
        return kwargs["cfg"]
    if args and isinstance(args[0], DictConfig):
        return args[0]
    raise ValueError("No config found in arguments or kwargs.")


def managed_run(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Manage a run."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cfg = _get_config(args, kwargs)
        print(f"Starting managed run with experiment: {cfg}")

        return fn(*args, **kwargs)

    return wrapper
