import logging
import random
import subprocess
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import hydra
import jax
import jax.numpy as jnp
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn import Module as PytorchModel

from simplexity.configs.generative_process.config import Config as GenerativeProcessConfig
from simplexity.configs.logging.config import Config as LoggingConfig
from simplexity.configs.mlflow.config import Config as MLFlowConfig
from simplexity.configs.persistence.config import Config as PersisterConfig
from simplexity.configs.predictive_model.config import HookedTransformerConfig, is_hooked_transformer_config
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.configs.training.optimizer.config import is_pytorch_optimizer_config
from simplexity.generative_processes.generative_process import GenerativeProcess
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
from simplexity.utils.mlflow_utils import get_experiment_id, resolve_registry_uri
from simplexity.utils.pytorch_utils import resolve_device

REQUIRED_TAGS = ["research_step", "retention"]


@dataclass
class Components:
    """Components for the run."""

    logger: Logger | None
    generative_process: GenerativeProcess | None
    initial_state: jax.Array | None
    persister: ModelPersister | None
    predictive_model: Any  # TODO: improve typing
    optimizer: Any | None  # TODO: improve typing


def _resolve_generative_process_config(generative_process_config: GenerativeProcessConfig) -> None:
    """Resolve the GenerativeProcessConfig."""
    vocab_size = generative_process_config.vocab_size
    if OmegaConf.is_missing(generative_process_config, "bos_token"):
        generative_process_config.bos_token = vocab_size
        vocab_size += 1
    if OmegaConf.is_missing(generative_process_config, "eos_token"):
        generative_process_config.eos_token = vocab_size


def _resolve_hooked_transformer_config(
    hooked_transformer_config: HookedTransformerConfig, generative_process_config: GenerativeProcessConfig
) -> None:
    """Resolve the HookedTransformerConfig."""
    base_vocab_size = generative_process_config.vocab_size
    use_bos = generative_process_config.bos_token is not None
    use_eos = generative_process_config.eos_token is not None
    d_vocab = base_vocab_size + int(use_bos) + int(use_eos)
    hooked_transformer_config.cfg.d_vocab = d_vocab
    hooked_transformer_config.cfg.device = resolve_device(hooked_transformer_config.cfg.device)


def _resolve_training_config(
    training_config: TrainingConfig,
    generative_process_config: GenerativeProcessConfig,
    hooked_transformer_config: HookedTransformerConfig,
) -> None:
    """Resolve the TrainingConfig."""
    if OmegaConf.is_missing(training_config, "sequence_len"):
        use_bos = generative_process_config.bos_token is not None
        use_eos = generative_process_config.eos_token is not None
        n_ctx = hooked_transformer_config.cfg.n_ctx
        sequence_len = n_ctx + 1 - int(use_bos) - int(use_eos)
        training_config.sequence_len = sequence_len


def _dynamic_resolve(cfg: DictConfig) -> None:
    generative_process_config: GenerativeProcessConfig | None = cfg.get("generative_process", None)
    if generative_process_config:
        _resolve_generative_process_config(generative_process_config)
        predictive_model_instance_config: DictConfig | None = OmegaConf.select(cfg, "predictive_model.instance")
        if predictive_model_instance_config and is_hooked_transformer_config(predictive_model_instance_config):
            hooked_transformer_config = cast(HookedTransformerConfig, predictive_model_instance_config)
            _resolve_hooked_transformer_config(hooked_transformer_config, generative_process_config)
            training_config: TrainingConfig | None = cfg.get("training", None)
            if training_config:
                _resolve_training_config(training_config, generative_process_config, hooked_transformer_config)


def _get_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DictConfig:
    """Get the config from the arguments."""
    if kwargs and "cfg" in kwargs:
        cfg = kwargs["cfg"]
    elif args and isinstance(args[0], DictConfig):
        cfg = args[0]
    else:
        raise ValueError("No config found in arguments or kwargs.")
    with open_dict(cfg):
        _dynamic_resolve(cfg)
    # TODO: validate the config
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    return cfg


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


def _setup_mlflow(cfg: DictConfig) -> mlflow.ActiveRun | nullcontext:
    """Setup the MLflow."""
    mlflow_config: MLFlowConfig | None = cfg.get("mlflow", None)
    if mlflow_config:
        if mlflow_config.tracking_uri:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        resolved_registry_uri = resolve_registry_uri(
            registry_uri=mlflow_config.registry_uri,
            tracking_uri=mlflow_config.tracking_uri,
            downgrade_unity_catalog=mlflow_config.downgrade_unity_catalog,
        )
        if resolved_registry_uri:
            mlflow.set_registry_uri(mlflow_config.registry_uri)
        experiment_id = get_experiment_id(mlflow_config.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_name = '{mlflow_config.run_name}'",
            max_results=1,
            output_format="list",
        )
        assert isinstance(runs, list)
        run_id = runs[0].info.run_id if runs else None
        return mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=mlflow_config.run_name,
            log_system_metrics=True,
        )
    return nullcontext()


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


def _setup_generative_process(cfg: DictConfig) -> GenerativeProcess | None:
    """Setup the generative process."""
    generative_process_config: GenerativeProcessConfig | None = cfg.get("generative_process", None)
    if generative_process_config:
        return typed_instantiate(generative_process_config.instance, GenerativeProcess)
    return None


def _setup_initial_state(cfg: DictConfig, generative_process: GenerativeProcess | None) -> jax.Array | None:
    """Setup the initial state."""
    if generative_process:
        batch_size = OmegaConf.select(cfg, "training.batch_size", default=1)
        return jnp.repeat(generative_process.initial_state[None, :], batch_size, axis=0)
    return None


def _setup_persister(cfg: DictConfig) -> ModelPersister | None:
    """Setup the persister."""
    persister_config: PersisterConfig | None = cfg.get("persistence", None)
    if persister_config:
        return typed_instantiate(persister_config.instance, ModelPersister)
    return None


def _setup_predictive_model(cfg: DictConfig, persister: ModelPersister | None) -> Any | None:
    """Setup the predictive model."""
    model: Any | None = None
    predictive_model_config: DictConfig | None = cfg.get("predictive_model", None)
    if predictive_model_config:
        instance_config = predictive_model_config.get("instance", None)
        if instance_config:
            model = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
        load_checkpoint_step = predictive_model_config.get("load_checkpoint_step", None)
        if load_checkpoint_step and persister:
            # model = persister.load_pytorch_model(load_checkpoint_step)
            pass
    return model


def _setup_optimizer(cfg: DictConfig, predictive_model: Any | None) -> Any | None:
    """Setup the optimizer."""
    optimizer_config: OptimizerConfig | None = OmegaConf.select(cfg, "training.optimizer", default=None)
    if optimizer_config:
        optimizer_instance_config: DictConfig = OmegaConf.select(cfg, "training.optimizer.instance")
        if is_pytorch_optimizer_config(optimizer_instance_config):
            if isinstance(predictive_model, PytorchModel):
                return hydra.utils.instantiate(
                    optimizer_config.instance, params=predictive_model.parameters()
                )  # TODO: cast to OptimizerConfig
            else:
                raise ValueError("Predictive model has no parameters")
        return hydra.utils.instantiate(optimizer_config.instance)  # TODO: typed instantiate
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
    generative_process = _setup_generative_process(cfg)
    initial_state = _setup_initial_state(cfg, generative_process)
    persister = _setup_persister(cfg)
    predictive_model = _setup_predictive_model(cfg, persister)
    optimizer = _setup_optimizer(cfg, predictive_model)
    return Components(
        logger=logger,
        generative_process=generative_process,
        initial_state=initial_state,
        persister=persister,
        predictive_model=predictive_model,
        optimizer=optimizer,
    )


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
            with _setup_mlflow(cfg):
                components = _setup(cfg, strict=strict, verbose=verbose)
                output = fn(*args, **kwargs, components=components)
                _cleanup(components)
            return output

        return wrapper

    return decorator
