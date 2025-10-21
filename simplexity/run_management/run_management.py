import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from simplexity.logging.logger import Logger
from simplexity.utils.hydra import typed_instantiate


@dataclass
class Components:
    """Components for the run."""

    logger: Logger | None


def _get_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DictConfig:
    """Get the config from the arguments."""
    if kwargs and "cfg" in kwargs:
        return kwargs["cfg"]
    if args and isinstance(args[0], DictConfig):
        return args[0]
    raise ValueError("No config found in arguments or kwargs.")


def _setup_logging(cfg: DictConfig) -> Logger | None:
    """Setup the logging."""
    # Suppress databricks SDK INFO messages
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    if cfg.logging and cfg.logging.instance:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        logger.log_config(cfg)
        logger.log_params(cfg)
        return logger
    return None


def _setup(cfg: DictConfig) -> Components:
    """Setup the run."""
    logger = _setup_logging(cfg)
    return Components(logger=logger)


def _cleanup(components: Components) -> None:
    """Cleanup the run."""
    if components.logger:
        components.logger.close()


def managed_run(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Manage a run."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cfg = _get_config(args, kwargs)
        components = _setup(cfg)
        output = fn(*args, **kwargs, components=components)
        _cleanup(components)
        return output

    return wrapper
