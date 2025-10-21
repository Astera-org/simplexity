import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from simplexity.logging.logger import Logger
from simplexity.run_management.environment_logging import (
    log_environment_artifacts,
    log_git_info,
    log_hydra_artifacts,
    log_source_script,
    log_system_info,
)
from simplexity.utils.hydra import typed_instantiate

REQUIRED_TAGS = ["research_step", "retention"]


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


def _working_tree_is_clean() -> bool:
    """Check if the working tree is clean."""
    result = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], capture_output=True, text=True)
    return result.returncode == 0


def _setup_logging(cfg: DictConfig) -> Logger | None:
    """Setup the logging."""
    # Suppress databricks SDK INFO messages
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    if cfg.logging and cfg.logging.instance:
        logger = typed_instantiate(cfg.logging.instance, Logger)
        return logger
    return None


def _do_logging(cfg: DictConfig, logger: Logger, verbose: bool) -> None:
    logger.log_config(cfg, resolve=True)
    logger.log_params(cfg)
    log_git_info(logger)
    log_system_info(logger)
    if cfg.tags:
        logger.log_tags(cfg.tags)
    if verbose:
        log_hydra_artifacts(logger)
        log_environment_artifacts(logger)
        log_source_script(logger)


def _setup(cfg: DictConfig, strict: bool, verbose: bool) -> Components:
    """Setup the run."""
    if strict:
        assert _working_tree_is_clean(), "Working tree is dirty"
        missing_required_tags = set(REQUIRED_TAGS) - set(cfg.tags)
        assert not missing_required_tags, "Tags must include " + ", ".join(missing_required_tags)
    logger = _setup_logging(cfg)
    if logger:
        _do_logging(cfg, logger, verbose)
    elif strict:
        raise ValueError("No logger found")
    return Components(logger=logger)


def _cleanup(components: Components) -> None:
    """Cleanup the run."""
    if components.logger:
        components.logger.close()


def managed_run(strict: bool = True, verbose: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Manage a run."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cfg = _get_config(args, kwargs)
            components = _setup(cfg, strict=strict, verbose=verbose)
            output = fn(*args, **kwargs, components=components)
            _cleanup(components)
            return output

        return wrapper

    return decorator
