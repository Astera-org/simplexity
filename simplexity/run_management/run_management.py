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


def _setup(cfg: DictConfig) -> None:
    """Setup the run."""
    print(f"Managed run setup with config: {cfg}")


def _cleanup(cfg: DictConfig) -> None:
    """Cleanup the run."""
    print(f"Managed run cleanup with config: {cfg}")


def managed_run(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Manage a run."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cfg = _get_config(args, kwargs)
        _setup(cfg)
        output = fn(*args, **kwargs)
        _cleanup(cfg)
        return output

    return wrapper
