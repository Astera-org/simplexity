import logging
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import hydra
from omegaconf import DictConfig

from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.persistence.config import Config as PersisterConfig
from simplexity.logging.logger import Logger
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.run_management.run_logging import (
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
    persister: ModelPersister | None
    predictive_model: Any  # TODO: improve typing


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


def _set_random_seeds(seed: int | None) -> None:
    """Seed available random number generators."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _setup_logging(cfg: DictConfig) -> Logger | None:
    """Setup the logging."""
    # Suppress databricks SDK INFO messages
    logging.getLogger("databricks.sdk").setLevel(logging.WARNING)
    logging_config: LoggingConfig | None = cfg.get("logging", None)
    if logging_config:
        logger = typed_instantiate(logging_config.instance, Logger)
        return logger
    return None


def _do_logging(cfg: DictConfig, logger: Logger, verbose: bool) -> None:
    logger.log_config(cfg, resolve=True)
    logger.log_params(cfg)
    log_git_info(logger)
    log_system_info(logger)
    tags = cfg.get("tags", {})
    if tags:
        logger.log_tags(tags)
    if verbose:
        log_hydra_artifacts(logger)
        log_environment_artifacts(logger)
        log_source_script(logger)


def _setup_persister(cfg: DictConfig) -> ModelPersister | None:
    """Setup the persister."""
    persister_config: PersisterConfig | None = cfg.get("persistence", None)
    if persister_config:
        return typed_instantiate(persister_config.instance, ModelPersister)
    return None


def _setup_predictive_model(cfg: DictConfig) -> Any | None:
    """Setup the predictive model."""
    predictive_model_config = cfg.get("predictive_model", None)
    if predictive_model_config:
        return hydra.utils.instantiate(predictive_model_config.instance)  # TODO: typed instantiate
        # TODO: load checkpoint using persister
    return None


def _setup(cfg: DictConfig, strict: bool, verbose: bool) -> Components:
    """Setup the run."""
    if strict:
        assert _working_tree_is_clean(), "Working tree is dirty"
        assert cfg.get("seed", None) is not None, "Seed must be set"
        tags: dict[str, Any] = cfg.get("tags", {})
        missing_required_tags = set(REQUIRED_TAGS) - set(tags.keys())
        assert not missing_required_tags, "Tags must include " + ", ".join(missing_required_tags)
    _set_random_seeds(cfg.get("seed", None))
    logger = _setup_logging(cfg)
    if logger:
        if strict:
            assert isinstance(logger, MLFlowLogger), "Logger must be an instance of MLFlowLogger"
            assert logger.tracking_uri, "Tracking URI must be set"
            assert logger.tracking_uri.startswith("databricks"), "Tracking URI must start with 'databricks'"
        _do_logging(cfg, logger, verbose)
    elif strict:
        raise ValueError("No logger found")
    persister = _setup_persister(cfg)
    predictive_model = _setup_predictive_model(cfg)
    return Components(logger=logger, persister=persister, predictive_model=predictive_model)


def _cleanup(components: Components) -> None:
    """Cleanup the run."""
    if components.logger:
        components.logger.close()
    if components.persister:
        components.persister.cleanup()


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
