import logging
import os
import random
import subprocess
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
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
from simplexity.utils.hydra import get_targets, typed_instantiate
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
    predictive_model: Any | None = None  # TODO: improve typing
    optimizer: Any | None = None  # TODO: improve typing


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


def _resolve_generative_process_config(generative_process_config: GenerativeProcessConfig) -> None:
    """Resolve the GenerativeProcessConfig."""
    vocab_size = generative_process_config.vocab_size
    SIMPLEXITY_LOGGER.info(f"[generative process] Base vocab size: {vocab_size}")
    if OmegaConf.is_missing(generative_process_config, "bos_token"):
        generative_process_config.bos_token = vocab_size
        SIMPLEXITY_LOGGER.info(f"[generative process] BOS token resolved to: {generative_process_config.bos_token}")
        vocab_size += 1
    elif generative_process_config.bos_token is not None:
        SIMPLEXITY_LOGGER.info(f"[generative process] BOS token defined as: {generative_process_config.bos_token}")
    if OmegaConf.is_missing(generative_process_config, "eos_token"):
        generative_process_config.eos_token = vocab_size
        SIMPLEXITY_LOGGER.info(f"[generative process] EOS token resolved to: {generative_process_config.eos_token}")
        vocab_size += 1
    elif generative_process_config.eos_token is not None:
        SIMPLEXITY_LOGGER.info(f"[generative process] EOS token defined as: {generative_process_config.eos_token}")
    SIMPLEXITY_LOGGER.info(f"[generative process] Total vocab size: {vocab_size}")


def _resolve_hooked_transformer_config(
    hooked_transformer_config: HookedTransformerConfig, generative_process_config: GenerativeProcessConfig
) -> None:
    """Resolve the HookedTransformerConfig."""
    base_vocab_size = generative_process_config.vocab_size
    use_bos = generative_process_config.bos_token is not None
    use_eos = generative_process_config.eos_token is not None
    d_vocab = base_vocab_size + int(use_bos) + int(use_eos)
    hooked_transformer_config.cfg.d_vocab = d_vocab
    SIMPLEXITY_LOGGER.info(f"[predictive model] d_vocab resolved to: {d_vocab}")
    hooked_transformer_config.cfg.device = resolve_device(hooked_transformer_config.cfg.device)
    SIMPLEXITY_LOGGER.info(f"[predictive model] device resolved to: {hooked_transformer_config.cfg.device}")


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
        SIMPLEXITY_LOGGER.info(f"[training] sequence len resolved to: {sequence_len}")
    else:
        SIMPLEXITY_LOGGER.info(f"[training] sequence len defined as: {training_config.sequence_len}")


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


def _setup_logging(cfg: DictConfig, instance_key: str) -> Logger:
    """Setup the logging."""
    logging_instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if logging_instance_config:
        logger = typed_instantiate(logging_instance_config, Logger)
        SIMPLEXITY_LOGGER.info(f"[logging] instantiated logger: {logger.__class__.__name__}")
        return logger
    raise KeyError


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


def _setup_generative_process(cfg: DictConfig, instance_key: str) -> GenerativeProcess:
    """Setup the generative process."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        generative_process = typed_instantiate(instance_config, GenerativeProcess)
        SIMPLEXITY_LOGGER.info(
            f"[generative process] instantiated generative process: {generative_process.__class__.__name__}"
        )
        return generative_process
    raise KeyError


def _setup_initial_state(cfg: DictConfig, generative_process: GenerativeProcess) -> jax.Array:
    """Setup the initial state."""
    batch_size = OmegaConf.select(cfg, "training.batch_size", default=1)
    initial_state = jnp.repeat(generative_process.initial_state[None, :], batch_size, axis=0)
    SIMPLEXITY_LOGGER.info(f"[generative process] instantiated initial state with shape: {initial_state.shape}")
    return initial_state


def _setup_persister(cfg: DictConfig, instance_key: str) -> ModelPersister:
    """Setup the persister."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        persister = typed_instantiate(instance_config, ModelPersister)
        SIMPLEXITY_LOGGER.info(f"[persister] instantiated persister: {persister.__class__.__name__}")
        return persister
    raise KeyError


def _setup_predictive_model(cfg: DictConfig, persister: ModelPersister | None) -> Any | None:
    """Setup the predictive model."""
    model: Any | None = None
    predictive_model_config: DictConfig | None = cfg.get("predictive_model", None)
    if predictive_model_config:
        instance_config = predictive_model_config.get("instance", None)
        if instance_config:
            with _suppress_pydantic_field_attribute_warning():
                model = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
            SIMPLEXITY_LOGGER.info(f"[predictive model] instantiated predictive model: {model.__class__.__name__}")
        load_checkpoint_step = predictive_model_config.get("load_checkpoint_step", None)
        if load_checkpoint_step and persister:
            # model = persister.load_pytorch_model(load_checkpoint_step)  # TODO: load checkpoint
            SIMPLEXITY_LOGGER.info(f"[predictive model] loaded checkpoint step: {load_checkpoint_step}")
    else:
        SIMPLEXITY_LOGGER.info("[predictive model] no predictive model config found")
    return model


def _setup_optimizer(cfg: DictConfig, predictive_model: Any | None) -> Any | None:
    """Setup the optimizer."""
    optimizer_config: OptimizerConfig | None = OmegaConf.select(cfg, "training.optimizer", default=None)
    if optimizer_config:
        optimizer_instance_config: DictConfig = OmegaConf.select(cfg, "training.optimizer.instance")
        if is_pytorch_optimizer_config(optimizer_instance_config):
            if isinstance(predictive_model, PytorchModel):
                optimizer = hydra.utils.instantiate(
                    optimizer_config.instance, params=predictive_model.parameters()
                )  # TODO: cast to OptimizerConfig
                SIMPLEXITY_LOGGER.info(f"[optimizer] instantiated optimizer: {optimizer.__class__.__name__}")
                return optimizer
            else:
                raise ValueError("Predictive model has no parameters")
        optimizer = hydra.utils.instantiate(optimizer_config.instance)  # TODO: typed instantiate
        SIMPLEXITY_LOGGER.info(f"[optimizer] instantiated optimizer: {optimizer.__class__.__name__}")
        return optimizer
    SIMPLEXITY_LOGGER.info("[optimizer] no optimizer config found")
    return None


def _setup(cfg: DictConfig, strict: bool, verbose: bool) -> Components:
    """Setup the run."""
    _setup_environment()
    if strict:
        _uv_sync()
        assert _working_tree_is_clean(), "Working tree is dirty"
        assert cfg.get("seed", None) is not None, "Seed must be set"
        tags: dict[str, Any] = cfg.get("tags", {})
        missing_required_tags = set(REQUIRED_TAGS) - set(tags.keys())
        assert not missing_required_tags, "Tags must include " + ", ".join(missing_required_tags)
        lock_dependencies = os.environ.get("MLFLOW_LOCK_MODEL_DEPENDENCIES")
        assert lock_dependencies, "MLFLOW_LOCK_MODEL_DEPENDENCIES must be set"
        assert lock_dependencies == "true", "MLFLOW_LOCK_MODEL_DEPENDENCIES must be set to true"
    _set_random_seeds(cfg.get("seed", None))
    components = Components()
    targets = get_targets(cfg)
    logger_targets = [target for target in targets if target.startswith("simplexity.logging.")]
    if logger_targets:
        loggers = [_setup_logging(cfg, logger_target) for logger_target in logger_targets]
        if strict:
            mlflow_loggers = [logger for logger in loggers if isinstance(logger, MLFlowLogger)]
            assert mlflow_loggers, "Logger must be an instance of MLFlowLogger"
            assert any(
                logger.tracking_uri and logger.tracking_uri.startswith("databricks") for logger in mlflow_loggers
            ), "Tracking URI must start with 'databricks'"
        for logger in loggers:
            _do_logging(cfg, logger, verbose)
        components.loggers = loggers
    else:
        SIMPLEXITY_LOGGER.info("[logging] no logging configs found")
        if strict:
            raise ValueError(f"Config must contain 1 logger, {len(logger_targets)} found")
    generative_process_targets = [target for target in targets if target.startswith("simplexity.generative_process.")]
    if generative_process_targets:
        components.generative_processes = [
            _setup_generative_process(cfg, target) for target in generative_process_targets
        ]
        components.initial_states = [_setup_initial_state(cfg, process) for process in components.generative_processes]

    else:
        SIMPLEXITY_LOGGER.info("[generative process] no generative process configs found")
    persister_targets = [target for target in targets if target.startswith("simplexity.persister.")]
    if persister_targets:
        components.persisters = [_setup_persister(cfg, target) for target in persister_targets]
    else:
        SIMPLEXITY_LOGGER.info("[persister] no persister configs found")
    persister = None
    if components.persisters:
        if len(components.persisters) == 1:
            persister = components.persisters[0]
        else:
            SIMPLEXITY_LOGGER.warning("Multiple persisters found, any model checkpoint loading will be skipped")
    components.predictive_model = _setup_predictive_model(cfg, persister)
    components.optimizer = _setup_optimizer(cfg, components.predictive_model)
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
                cfg = _get_config(args, kwargs)
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
