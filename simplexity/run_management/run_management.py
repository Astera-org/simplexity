"""Run management utilities for orchestrating experiment setup and teardown.

This module centralizes environment setup, configuration resolution, component
instantiation (logging, generative processes, models, optimizers), MLflow run
management, and cleanup via the `managed_run` decorator.
"""

import logging
import os
import random
import subprocess
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import mlflow
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module as PytorchModel

from simplexity.configs.generative_process.config import Config as GenerativeProcessConfig
from simplexity.configs.mlflow.config import Config as MLFlowConfig
from simplexity.configs.predictive_model.config import HookedTransformerConfig, is_hooked_transformer_config
from simplexity.configs.training.config import Config as TrainingConfig
from simplexity.configs.training.optimizer.config import Config as OptimizerConfig
from simplexity.configs.training.optimizer.config import is_pytorch_optimizer_config
from simplexity.generative_processes.generative_process import GenerativeProcess, is_generative_process_target
from simplexity.logging.logger import Logger, is_logger_target
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.persistence.model_persister import ModelPersister, is_model_persister_target
from simplexity.predictive_models.predictive_model import is_predictive_model_target
from simplexity.run_management.run_logging import (
    log_environment_artifacts,
    log_git_info,
    log_hydra_artifacts,
    log_source_script,
    log_system_info,
)
from simplexity.training.optimizer import is_optimizer_target
from simplexity.utils.config_utils import (
    dynamic_resolve,
    filter_instance_keys,
    get_config,
    get_instance_keys,
    typed_instantiate,
)
from simplexity.utils.mlflow_utils import get_experiment_id, resolve_registry_uri
from simplexity.utils.pytorch_utils import resolve_device

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")
logging.captureWarnings(True)

DEFAULT_ENVIRONMNENT_VARIABLES = {
    "MLFLOW_LOCK_MODEL_DEPENDENCIES": "true",
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
}
REQUIRED_TAGS = ["research_step", "retention"]


@dataclass
class Components:
    """Components for the run."""

    loggers: list[Logger] | None = None
    generative_processes: list[GenerativeProcess] | None = None
    initial_states: list[jax.Array] | None = None
    persisters: list[ModelPersister] | None = None
    predictive_models: list[Any] | None = None  # TODO: improve typing
    optimizers: list[Any] | None = None  # TODO: improve typing


@contextmanager
def _suppress_pydantic_field_attribute_warning() -> Iterator[None]:
    """Temporarily ignore noisy Pydantic field attribute warnings from dependencies.

    If Hydra instantiates a HookedTransformer, it imports transformer_lens, which in turn imports W&B (wandb).
    As soon as W&B loads, it builds a large set of Pydantic models (for example in wandb/automations/automations.py).
    Those models declare fields like:

    ```python
    created_at: Annotated[datetime, Field(repr=False, frozen=True, alias="createdAt")]
    ```

    Pydantic v2 interprets those Field(...) arguments, spots repr=False and frozen=True,
    and issues UnsupportedFieldAttributeWarning because those keywords are only meaningful for dataclass fields,
    they have no effect on a BaseModel.
    """
    try:
        from pydantic.warnings import UnsupportedFieldAttributeWarning
    except ModuleNotFoundError:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
        yield


def _setup_environment() -> None:
    """Setup the environment."""
    for key, value in DEFAULT_ENVIRONMNENT_VARIABLES.items():
        if not os.environ.get(key):
            os.environ[key] = value
            SIMPLEXITY_LOGGER.info(f"[environment] {key} set to: {os.environ[key]}")
        else:
            SIMPLEXITY_LOGGER.info(f"[environment] {key} already set to: {os.environ[key]}")


def _uv_sync() -> None:
    """Sync the uv environment."""
    args = ["uv", "sync", "--extra", "pytorch"]
    device = resolve_device()
    if device == "cuda":
        args.extend(["--extra", "cuda"])
    elif device == "mps":
        args.extend(["--extra", "mac"])
    subprocess.run(args, check=True)


def _working_tree_is_clean() -> bool:
    """Check if the working tree is clean."""
    result = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], capture_output=True, text=True)
    return result.returncode == 0


def _set_random_seeds(seed: int | None) -> None:
    """Seed available random number generators."""
    if seed is None:
        return
    random.seed(seed)
    SIMPLEXITY_LOGGER.info(f"[random] seed set to: {seed}")
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
        SIMPLEXITY_LOGGER.info(f"[numpy] seed set to: {seed}")
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        SIMPLEXITY_LOGGER.info(f"[torch] seed set to: {seed}")
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            SIMPLEXITY_LOGGER.info(f"[torch] CUDA seed set to: {seed}")


def _assert_reproducibile(cfg: DictConfig) -> None:
    assert _working_tree_is_clean(), "Working tree is dirty"
    assert cfg.get("seed", None) is not None, "Seed must be set"
    lock_dependencies = os.environ.get("MLFLOW_LOCK_MODEL_DEPENDENCIES")
    assert lock_dependencies, "MLFLOW_LOCK_MODEL_DEPENDENCIES must be set"
    assert lock_dependencies == "true", "MLFLOW_LOCK_MODEL_DEPENDENCIES must be set to true"


def _assert_tagged(cfg: DictConfig) -> None:
    tags: dict[str, Any] = cfg.get("tags", {})
    missing_required_tags = set(REQUIRED_TAGS) - set(tags.keys())
    assert not missing_required_tags, "Tags must include " + ", ".join(missing_required_tags)


def _setup_mlflow(cfg: DictConfig) -> mlflow.ActiveRun | nullcontext:
    """Setup the MLflow."""
    mlflow_config: MLFlowConfig | None = cfg.get("mlflow", None)
    if mlflow_config:
        if mlflow_config.tracking_uri:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            SIMPLEXITY_LOGGER.info(f"[mlflow] tracking uri: {mlflow.get_tracking_uri()}")
        resolved_registry_uri = resolve_registry_uri(
            registry_uri=mlflow_config.registry_uri,
            tracking_uri=mlflow_config.tracking_uri,
            downgrade_unity_catalog=mlflow_config.downgrade_unity_catalog,
        )
        if resolved_registry_uri:
            mlflow.set_registry_uri(mlflow_config.registry_uri)
            SIMPLEXITY_LOGGER.info(f"[mlflow] registry uri: {mlflow.get_registry_uri()}")
        experiment_id = get_experiment_id(mlflow_config.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_name = '{mlflow_config.run_name}'",
            max_results=1,
            output_format="list",
        )
        assert isinstance(runs, list)
        if runs:
            run_id = runs[0].info.run_id
            SIMPLEXITY_LOGGER.info(
                f"[mlflow] run with name '{mlflow_config.run_name}' already exists with id: {run_id}"
            )
        else:
            run_id = None
            SIMPLEXITY_LOGGER.info(f"[mlflow] run with name '{mlflow_config.run_name}' does not yet exist")
        SIMPLEXITY_LOGGER.info(
            f"[mlflow] starting run with: "
            f"{{id: {run_id}, experiment id: {experiment_id}, run name: {mlflow_config.run_name}}}"
        )
        return mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=mlflow_config.run_name,
            log_system_metrics=True,
        )
    return nullcontext()


def _instantiate_logger(cfg: DictConfig, instance_key: str) -> Logger:
    """Setup the logging."""
    logging_instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if logging_instance_config:
        logger = typed_instantiate(logging_instance_config, Logger)
        SIMPLEXITY_LOGGER.info(f"[logging] instantiated logger: {logger.__class__.__name__}")
        return logger
    raise KeyError


def _setup_logging(cfg: DictConfig, instance_keys: list[str], *, strict: bool) -> list[Logger] | None:
    instance_keys = filter_instance_keys(cfg, instance_keys, is_logger_target)
    if instance_keys:
        loggers = [_instantiate_logger(cfg, instance_key) for instance_key in instance_keys]
        if strict:
            mlflow_loggers = [logger for logger in loggers if isinstance(logger, MLFlowLogger)]
            assert mlflow_loggers, "Logger must be an instance of MLFlowLogger"
            assert any(
                logger.tracking_uri and logger.tracking_uri.startswith("databricks") for logger in mlflow_loggers
            ), "Tracking URI must start with 'databricks'"
        return loggers
    SIMPLEXITY_LOGGER.info("[logging] no logging configs found")
    if strict:
        raise ValueError(f"Config must contain 1 logger, {len(instance_keys)} found")
    return None


def _instantiate_generative_process(cfg: DictConfig, instance_key: str) -> GenerativeProcess:
    """Setup the generative process."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        generative_process = typed_instantiate(instance_config, GenerativeProcess)
        SIMPLEXITY_LOGGER.info(
            f"[generative process] instantiated generative process: {generative_process.__class__.__name__}"
        )
        return generative_process
    raise KeyError


def _create_initial_state(cfg: DictConfig, generative_process: GenerativeProcess) -> jax.Array:
    """Setup the initial state."""
    batch_size = OmegaConf.select(cfg, "training.batch_size", default=1)
    initial_state = jnp.repeat(generative_process.initial_state[None, :], batch_size, axis=0)
    SIMPLEXITY_LOGGER.info(f"[generative process] instantiated initial state with shape: {initial_state.shape}")
    return initial_state


@dynamic_resolve
def _resolve_generative_process_config(cfg: GenerativeProcessConfig, base_vocab_size: int) -> None:
    """Resolve the GenerativeProcessConfig."""
    cfg.vocab_size = base_vocab_size
    SIMPLEXITY_LOGGER.info(f"[generative process] Base vocab size: {base_vocab_size}")
    vocab_size = base_vocab_size
    if OmegaConf.is_missing(cfg, "bos_token"):
        cfg.bos_token = vocab_size
        SIMPLEXITY_LOGGER.info(f"[generative process] BOS token resolved to: {cfg.bos_token}")
        vocab_size += 1
    elif cfg.bos_token is not None:
        SIMPLEXITY_LOGGER.info(f"[generative process] BOS token defined as: {cfg.bos_token}")
    if OmegaConf.is_missing(cfg, "eos_token"):
        cfg.eos_token = vocab_size
        SIMPLEXITY_LOGGER.info(f"[generative process] EOS token resolved to: {cfg.eos_token}")
        vocab_size += 1
    elif cfg.eos_token is not None:
        SIMPLEXITY_LOGGER.info(f"[generative process] EOS token defined as: {cfg.eos_token}")
    SIMPLEXITY_LOGGER.info(f"[generative process] Total vocab size: {vocab_size}")


def _setup_generative_processes(
    cfg: DictConfig, instance_keys: list[str]
) -> tuple[list[GenerativeProcess] | None, list[jax.Array] | None]:
    instance_keys = filter_instance_keys(cfg, instance_keys, is_generative_process_target)
    if instance_keys:
        generative_processes = []
        for instance_key in instance_keys:
            generative_process = _instantiate_generative_process(cfg, instance_key)
            config_key = instance_key.rsplit(".", 1)[0]
            generative_process_config: GenerativeProcessConfig | None = OmegaConf.select(cfg, config_key)
            if generative_process_config is None:
                raise RuntimeError("Error selecting generative process config")
            base_vocab_size = generative_process.vocab_size
            _resolve_generative_process_config(generative_process_config, base_vocab_size)
            generative_processes.append(generative_process)
        initial_states = [_create_initial_state(cfg, process) for process in generative_processes]
        return generative_processes, initial_states
    SIMPLEXITY_LOGGER.info("[generative process] no generative process configs found")
    return None, None


def _instantiate_persister(cfg: DictConfig, instance_key: str) -> ModelPersister:
    """Setup the persister."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        persister = typed_instantiate(instance_config, ModelPersister)
        SIMPLEXITY_LOGGER.info(f"[persister] instantiated persister: {persister.__class__.__name__}")
        return persister
    raise KeyError


def _setup_persisters(cfg: DictConfig, instance_keys: list[str]) -> list[ModelPersister] | None:
    instance_keys = filter_instance_keys(cfg, instance_keys, is_model_persister_target)
    if instance_keys:
        return [_instantiate_persister(cfg, instance_key) for instance_key in instance_keys]
    SIMPLEXITY_LOGGER.info("[persister] no persister configs found")
    return None


def _get_persister(persisters: list[ModelPersister] | None) -> ModelPersister | None:
    if persisters:
        if len(persisters) == 1:
            return persisters[0]
        SIMPLEXITY_LOGGER.warning("Multiple persisters found, any model model checkpoint loading will be skipped")
        return None
    SIMPLEXITY_LOGGER.warning("No persister found, any model checkpoint loading will be skipped")
    return None


@dynamic_resolve
def _resolve_hooked_transformer_config(cfg: HookedTransformerConfig, *, vocab_size: int) -> None:
    """Resolve the HookedTransformerConfig."""
    cfg.cfg.d_vocab = vocab_size
    SIMPLEXITY_LOGGER.info(f"[predictive model] d_vocab resolved to: {vocab_size}")
    cfg.cfg.device = resolve_device(cfg.cfg.device)
    SIMPLEXITY_LOGGER.info(f"[predictive model] device resolved to: {cfg.cfg.device}")


def _instantiate_predictive_model(cfg: DictConfig, instance_key: str) -> Any:
    """Setup the predictive model."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        with _suppress_pydantic_field_attribute_warning():
            predictive_model = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
        SIMPLEXITY_LOGGER.info(
            f"[predictive model] instantiated predictive model: {predictive_model.__class__.__name__}"
        )
        return predictive_model
    raise KeyError


def _load_checkpoint(cfg: DictConfig, target: str, persisters: list[ModelPersister] | None) -> None:
    """Load the checkpoint."""
    load_checkpoint_step = cfg.get("load_checkpoint_step", None)
    if load_checkpoint_step:
        persister = _get_persister(persisters)
        if persister:
            # model = persister.load_pytorch_model(load_checkpoint_step)  # TODO: load checkpoint
            SIMPLEXITY_LOGGER.info(f"[predictive model] loaded checkpoint step: {load_checkpoint_step}")
        else:
            raise RuntimeError("Unable to load model checkpoint")


def _setup_predictive_models(
    cfg: DictConfig, instance_keys: list[str], persisters: list[ModelPersister] | None
) -> list[Any] | None:
    """Setup the predictive model."""
    models = []
    instance_keys = filter_instance_keys(cfg, instance_keys, is_predictive_model_target)
    for instance_key in instance_keys:
        instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
        if instance_config and is_hooked_transformer_config(instance_config):
            _resolve_hooked_transformer_config(
                instance_config, vocab_size=4
            )  # TODO: get vocab size from generative processes
        models.append(_instantiate_predictive_model(cfg, instance_key))
        _load_checkpoint(cfg, instance_key, persisters)
    if models:
        return models
    SIMPLEXITY_LOGGER.info("[predictive model] no predictive model config found")
    return None


def _get_predictive_model(predictive_models: list[Any] | None) -> Any | None:
    if predictive_models:
        if len(predictive_models) == 1:
            return predictive_models[0]
        SIMPLEXITY_LOGGER.warning("Multiple predictive models found, any model checkpoint loading will be skipped")
        return None
    SIMPLEXITY_LOGGER.warning("No predictive model found, any model checkpoint loading will be skipped")
    return None


def _instantiate_optimizer(cfg: DictConfig, instance_key: str, predictive_model: Any | None) -> Any:
    """Setup the optimizer."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        optimizer_instance_config: DictConfig = OmegaConf.select(cfg, "training.optimizer.instance")
        if is_pytorch_optimizer_config(optimizer_instance_config):
            if predictive_model and isinstance(predictive_model, PytorchModel):
                optimizer = hydra.utils.instantiate(instance_config, params=predictive_model.parameters())
                SIMPLEXITY_LOGGER.info(f"[optimizer] instantiated optimizer: {optimizer.__class__.__name__}")
                return optimizer
            SIMPLEXITY_LOGGER.warning("Predictive model has no parameters, optimizer will be skipped")
            return None
        optimizer = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
        SIMPLEXITY_LOGGER.info(f"[optimizer] instantiated optimizer: {optimizer.__class__.__name__}")
        return optimizer
    raise KeyError


def _setup_optimizers(
    cfg: DictConfig, instance_keys: list[str], predictive_models: list[Any] | None
) -> list[Any] | None:
    """Setup the optimizer."""
    instance_keys = filter_instance_keys(cfg, instance_keys, is_optimizer_target)
    if instance_keys:
        model = _get_predictive_model(predictive_models)
        return [_instantiate_optimizer(cfg, instance_key, model) for instance_key in instance_keys]
    SIMPLEXITY_LOGGER.info("[optimizer] no optimizer configs found")
    return None


def _get_special_token(cfg: DictConfig, instance_keys: list[str], token: str) -> int | None:
    instance_keys = filter_instance_keys(cfg, instance_keys, is_generative_process_target)
    token_value: int | None = None
    for instance_key in instance_keys:
        config_key = instance_key.rsplit(".", 1)[0]
        generative_process_config: DictConfig | None = OmegaConf.select(cfg, config_key, throw_on_missing=True)
        if generative_process_config is None:
            raise RuntimeError("Error selecting generative process config")
        new_token_value: int = generative_process_config.get(f"{token}_token")
        if token_value is None:
            token_value = new_token_value
        elif new_token_value != token_value:
            SIMPLEXITY_LOGGER.warning(
                f"[generative process] Multiple generative processes with conflicting {token} token values"
            )
            return None
    return token_value


@dynamic_resolve
def _resolve_training_config(cfg: TrainingConfig, *, n_ctx: int, use_bos: bool, use_eos: bool) -> None:
    """Resolve the TrainingConfig."""
    if OmegaConf.is_missing(cfg, "sequence_len"):
        sequence_len = n_ctx + 1 - int(use_bos) - int(use_eos)
        cfg.sequence_len = sequence_len
        SIMPLEXITY_LOGGER.info(f"[training] sequence len resolved to: {sequence_len}")
    else:
        SIMPLEXITY_LOGGER.info(f"[training] sequence len defined as: {cfg.sequence_len}")


def _setup_training(cfg: DictConfig, instance_keys: list[str]) -> None:
    training_config: TrainingConfig | None = cfg.get("training", None)
    if training_config:
        n_ctx: int = OmegaConf.select(
            cfg,
            "predictive_model.instance.cfg.n_ctx",
        )
        use_bos = _get_special_token(cfg, instance_keys, "bos") is not None
        use_eos = _get_special_token(cfg, instance_keys, "eos") is not None
        _resolve_training_config(training_config, n_ctx=n_ctx, use_bos=use_bos, use_eos=use_eos)


def _do_logging(cfg: DictConfig, loggers: list[Logger] | None, verbose: bool) -> None:
    if loggers is None:
        return
    for logger in loggers:
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


def _setup(cfg: DictConfig, strict: bool, verbose: bool) -> Components:
    """Setup the run."""
    _setup_environment()
    if strict:
        _uv_sync()
        _assert_reproducibile(cfg)
        _assert_tagged(cfg)
    _set_random_seeds(cfg.get("seed", None))
    components = Components()
    instance_keys = get_instance_keys(cfg)
    components.loggers = _setup_logging(cfg, instance_keys, strict=strict)
    generative_processes, initial_states = _setup_generative_processes(cfg, instance_keys)
    components.generative_processes = generative_processes
    components.initial_states = initial_states
    components.persisters = _setup_persisters(cfg, instance_keys)
    components.predictive_models = _setup_predictive_models(cfg, instance_keys, components.persisters)
    components.optimizers = _setup_optimizers(cfg, instance_keys, components.predictive_models)
    _setup_training(cfg, instance_keys)
    _do_logging(cfg, components.loggers, verbose)
    return components


def _cleanup(components: Components) -> None:
    """Cleanup the run."""
    if components.loggers:
        for logger in components.loggers:
            logger.close()
    if components.persisters:
        for persister in components.persisters:
            persister.cleanup()


def managed_run(strict: bool = True, verbose: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Manage a run."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                cfg = get_config(args, kwargs)
                # TODO: validate the config
                with _setup_mlflow(cfg):
                    components = _setup(cfg, strict=strict, verbose=verbose)
                    output = fn(*args, **kwargs, components=components)
                _cleanup(components)
                return output
            except Exception as e:
                SIMPLEXITY_LOGGER.error(f"[run] error: {e}")
                raise e

        return wrapper

    return decorator
