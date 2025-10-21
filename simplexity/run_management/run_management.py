import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from simplexity.logging.logger import Logger
from simplexity.run_management.environment_logging import log_git_info
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


_HYDRA_ARTIFACTS = ("config.yaml", "hydra.yaml", "overrides.yaml")


def _log_hydra_artifacts(logger: Logger) -> None:
    try:
        hydra_dir = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
    except Exception:
        return
    for artifact in _HYDRA_ARTIFACTS:
        path = hydra_dir / artifact
        if path.exists():
            try:
                logger.log_artifact(str(path), artifact_path=".hydra")
            except Exception as e:
                logging.warning("Failed to log Hydra artifact %s: %s", path, e)


def _setup(cfg: DictConfig, strict: bool, verbose: bool) -> Components:
    """Setup the run."""
    if strict:
        assert _working_tree_is_clean(), "Working tree is dirty"
    logger = _setup_logging(cfg)
    if logger:
        logger.log_config(cfg, resolve=True)
        logger.log_params(cfg)
        log_git_info(logger)
        if verbose:
            _log_hydra_artifacts(logger)
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
