"""Run management utilities for orchestrating experiment setup and teardown.

This module centralizes environment setup, configuration resolution, component
instantiation (logging, generative processes, models, optimizers), MLflow run
management, and cleanup via the `managed_run` decorator.
"""

import logging
import os
import random
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn import Module as PytorchModel

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.logging.mlflow_logger import MLFlowLogger
from simplexity.persistence.model_persister import ModelPersister
from simplexity.run_management.components import Components
from simplexity.run_management.run_logging import (
    log_environment_artifacts,
    log_git_info,
    log_hydra_artifacts,
    log_source_script,
    log_system_info,
)
from simplexity.run_management.structured_configs import (
    is_generative_process_target,
    is_hooked_transformer_config,
    is_logger_target,
    is_model_persister_target,
    is_optimizer_target,
    is_predictive_model_target,
    is_pytorch_optimizer_config,
    resolve_generative_process_config,
    resolve_hooked_transformer_config,
    validate_base_config,
    validate_generative_process_config,
    validate_logging_config,
    validate_optimizer_config,
    validate_persistence_config,
)
from simplexity.utils.config_utils import (
    filter_instance_keys,
    get_config,
    get_instance_keys,
    typed_instantiate,
)
from simplexity.utils.mlflow_utils import get_experiment_id, resolve_registry_uri
from simplexity.utils.pytorch_utils import resolve_device

SIMPLEXITY_LOGGER = logging.getLogger("simplexity")
logging.captureWarnings(True)

DEFAULT_ENVIRONMENT_VARIABLES = {
    "MLFLOW_LOCK_MODEL_DEPENDENCIES": "true",
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
}
REQUIRED_TAGS = ["research_step", "retention"]


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


def _load_config(cfg: DictConfig, load_config: DictConfig) -> None:
    """Load the config."""
    if not load_config:
        SIMPLEXITY_LOGGER.warning("[config] load_config entry is empty, skipping")
        return

    tracking_uri: str | None = load_config.get("tracking_uri")
    experiment_name: str | None = load_config.get("experiment_name")
    run_name: str | None = load_config.get("run_name")
    configs_to_load: DictConfig | None = load_config.get("configs")
    artifact_path: str = load_config.get("artifact_path", "config.yaml")

    if not experiment_name:
        raise ValueError("load_config.experiment_name is required")
    if not run_name:
        raise ValueError("load_config.run_name is required")
    if not configs_to_load:
        SIMPLEXITY_LOGGER.warning(f"[config] no configs specified for load_config run '{run_name}', nothing to merge")
        return

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found for load_config run '{run_name}'")
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"Run with name '{run_name}' not found in experiment '{experiment_name}' for load_config entry"
        )
    run = runs[0]
    run_id = run.info.run_id

    SIMPLEXITY_LOGGER.info(
        f"[config] loading artifact '{artifact_path}' from run '{run_name}' ({run_id}) "
        f"in experiment '{experiment_name}'"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_local_path = client.download_artifacts(run_id, artifact_path, temp_dir)
        source_cfg = OmegaConf.load(artifact_local_path)

    configs_mapping: dict[str, str] = OmegaConf.to_container(configs_to_load, resolve=True)  # type: ignore[arg-type]
    with open_dict(cfg):
        for source_key, destination_key in configs_mapping.items():
            if not isinstance(source_key, str) or not source_key:
                raise ValueError("load_config configs keys must be non-empty strings")
            if not isinstance(destination_key, str) or not destination_key:
                raise ValueError("load_config configs values must be non-empty strings")

            selected_config = OmegaConf.select(source_cfg, source_key, throw_on_missing=False)
            if selected_config is None:
                raise KeyError(f"Config key '{source_key}' not found in run '{run_name}' artifact '{artifact_path}'")

            cloned_config = OmegaConf.create(OmegaConf.to_container(selected_config, resolve=False))
            existing_destination = OmegaConf.select(cfg, destination_key, throw_on_missing=False)
            if existing_destination is None:
                SIMPLEXITY_LOGGER.info(
                    f"[config] adding config '{source_key}' from run '{run_name}' to '{destination_key}'"
                )
                OmegaConf.update(cfg, destination_key, cloned_config, force_add=True)
            else:
                SIMPLEXITY_LOGGER.info(
                    f"[config] merging config '{source_key}' from run '{run_name}' into '{destination_key}'"
                )
                merged_config = OmegaConf.merge(cloned_config, existing_destination)
                OmegaConf.update(cfg, destination_key, merged_config, force_add=True)


def _get_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DictConfig:
    """Get the config from the arguments."""
    cfg = get_config(args, kwargs)
    load_configs: list[DictConfig] = cfg.get("load_configs", [])
    for load_config in load_configs:
        _load_config(cfg, load_config)
    return cfg


def _setup_environment() -> None:
    """Setup the environment."""
    for key, value in DEFAULT_ENVIRONMENT_VARIABLES.items():
        if not os.environ.get(key):
            os.environ[key] = value
            SIMPLEXITY_LOGGER.info("[environment] %s set to: %s", key, os.environ[key])
        else:
            SIMPLEXITY_LOGGER.info("[environment] %s already set to: %s", key, os.environ[key])


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
    if seed is None:
        return
    random.seed(seed)
    SIMPLEXITY_LOGGER.info("[random] seed set to: %s", seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
        SIMPLEXITY_LOGGER.info("[numpy] seed set to: %s", seed)
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        SIMPLEXITY_LOGGER.info("[torch] seed set to: %s", seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            SIMPLEXITY_LOGGER.info("[torch] CUDA seed set to: %s", seed)


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


def _setup_mlflow(cfg: DictConfig) -> mlflow.ActiveRun | nullcontext[None]:
    mlflow_config: DictConfig | None = cfg.get("mlflow", None)
    if mlflow_config:
        experiment_name: str | None = mlflow_config.get("experiment_name", None)
        assert experiment_name is not None, "Experiment name must be set"
        run_name: str | None = mlflow_config.get("run_name", None)
        assert run_name is not None, "Run name must be set"
        tracking_uri: str | None = mlflow_config.get("tracking_uri", None)
        registry_uri: str | None = mlflow_config.get("registry_uri", None)
        downgrade_unity_catalog: bool = mlflow_config.get("downgrade_unity_catalog", True)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            SIMPLEXITY_LOGGER.info("[mlflow] tracking uri: %s", mlflow.get_tracking_uri())
        resolved_registry_uri = resolve_registry_uri(
            registry_uri=registry_uri,
            tracking_uri=tracking_uri,
            downgrade_unity_catalog=downgrade_unity_catalog,
        )
        if resolved_registry_uri:
            mlflow.set_registry_uri(resolved_registry_uri)
            SIMPLEXITY_LOGGER.info("[mlflow] registry uri: %s", mlflow.get_registry_uri())
        experiment_id = get_experiment_id(experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_name = '{run_name}'",
            max_results=1,
            output_format="list",
        )
        assert isinstance(runs, list)
        if runs:
            run_id = runs[0].info.run_id
            SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' already exists with id: %s", run_name, run_id)
        else:
            run_id = None
            SIMPLEXITY_LOGGER.info("[mlflow] run with name '%s' does not yet exist", run_name)
        SIMPLEXITY_LOGGER.info(
            "[mlflow] starting run with: {id: %s, experiment id: %s, run name: %s}", run_id, experiment_id, run_name
        )
        return mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            log_system_metrics=True,
        )
    return nullcontext()


def _instantiate_logger(cfg: DictConfig, instance_key: str) -> Logger:
    """Setup the logging."""
    logging_instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if logging_instance_config:
        logger = typed_instantiate(logging_instance_config, Logger)
        SIMPLEXITY_LOGGER.info("[logging] instantiated logger: %s", logger.__class__.__name__)
        return logger
    raise KeyError


def _setup_logging(cfg: DictConfig, instance_keys: list[str], *, strict: bool) -> dict[str, Logger] | None:
    instance_keys = filter_instance_keys(
        cfg,
        instance_keys,
        is_logger_target,
        validate_fn=validate_logging_config,
        component_name="logging",
    )
    if instance_keys:
        loggers = {instance_key: _instantiate_logger(cfg, instance_key) for instance_key in instance_keys}
        if strict:
            mlflow_loggers = [logger for logger in loggers.values() if isinstance(logger, MLFlowLogger)]
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
            "[generative process] instantiated generative process: %s", generative_process.__class__.__name__
        )
        return generative_process
    raise KeyError


def _create_initial_state(generative_process: GenerativeProcess, batch_size: int | None) -> jax.Array:
    """Setup the initial state."""
    if batch_size is None:
        initial_state = generative_process.initial_state
    else:
        initial_state = jnp.repeat(generative_process.initial_state[None, :], batch_size, axis=0)
    SIMPLEXITY_LOGGER.info("[generative process] instantiated initial state with shape: %s", initial_state.shape)
    return initial_state


def _setup_generative_processes(
    cfg: DictConfig, instance_keys: list[str]
) -> tuple[dict[str, GenerativeProcess] | None, dict[str, jax.Array] | None]:
    instance_keys = filter_instance_keys(
        cfg,
        instance_keys,
        is_generative_process_target,
        validate_fn=validate_generative_process_config,
        component_name="generative process",
    )
    if instance_keys:
        generative_processes = {}
        initial_states = {}
        for instance_key in instance_keys:
            generative_process = _instantiate_generative_process(cfg, instance_key)
            config_key = instance_key.rsplit(".", 1)[0]
            generative_process_config: DictConfig | None = OmegaConf.select(cfg, config_key)
            if generative_process_config is None:
                raise RuntimeError("Error selecting generative process config")
            base_vocab_size = generative_process.vocab_size
            resolve_generative_process_config(generative_process_config, base_vocab_size)
            generative_processes[instance_key] = generative_process
            batch_size = generative_process_config.get("batch_size", None)
            initial_states[instance_key] = _create_initial_state(generative_process, batch_size)
        return generative_processes, initial_states
    SIMPLEXITY_LOGGER.info("[generative process] no generative process configs found")
    return None, None


def _instantiate_persister(cfg: DictConfig, instance_key: str) -> ModelPersister:
    """Setup the persister."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        persister: ModelPersister = hydra.utils.instantiate(instance_config)
        SIMPLEXITY_LOGGER.info("[persister] instantiated persister: %s", persister.__class__.__name__)
        return persister
    raise KeyError


def _setup_persisters(cfg: DictConfig, instance_keys: list[str]) -> dict[str, ModelPersister] | None:
    instance_keys = filter_instance_keys(
        cfg,
        instance_keys,
        is_model_persister_target,
        validate_fn=validate_persistence_config,
        component_name="persistence",
    )
    if instance_keys:
        return {instance_key: _instantiate_persister(cfg, instance_key) for instance_key in instance_keys}
    SIMPLEXITY_LOGGER.info("[persister] no persister configs found")
    return None


def _get_persister(persisters: dict[str, ModelPersister] | None) -> ModelPersister | None:
    if persisters:
        if len(persisters) == 1:
            return next(iter(persisters.values()))
        SIMPLEXITY_LOGGER.warning("Multiple persisters found, any model model checkpoint loading will be skipped")
        return None
    SIMPLEXITY_LOGGER.warning("No persister found, any model checkpoint loading will be skipped")
    return None


def _get_attribute_value(cfg: DictConfig, instance_keys: list[str], attribute_name: str) -> int | None:
    """Get the vocab size."""
    instance_keys = filter_instance_keys(
        cfg,
        instance_keys,
        is_generative_process_target,
        validate_fn=validate_generative_process_config,
        component_name="generative process",
    )
    attribute_value: int | None = None
    for instance_key in instance_keys:
        config_key = instance_key.rsplit(".", 1)[0]
        generative_process_config: DictConfig | None = OmegaConf.select(cfg, config_key, throw_on_missing=True)
        if generative_process_config is None:
            raise RuntimeError("Error selecting generative process config")
        new_attribute_value: int | None = OmegaConf.select(
            generative_process_config, attribute_name, throw_on_missing=False, default=None
        )
        if attribute_value is None:
            attribute_value = new_attribute_value
        elif new_attribute_value != attribute_value:
            SIMPLEXITY_LOGGER.warning(
                f"[generative process] Multiple generative processes with conflicting {attribute_name} values"
            )
            return None
    return attribute_value


def _instantiate_predictive_model(cfg: DictConfig, instance_key: str) -> Any:
    """Setup the predictive model."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        with _suppress_pydantic_field_attribute_warning():
            predictive_model = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
        SIMPLEXITY_LOGGER.info(
            "[predictive model] instantiated predictive model: %s", predictive_model.__class__.__name__
        )
        return predictive_model
    raise KeyError


def _load_checkpoint(model: Any, persisters: dict[str, ModelPersister] | None, load_checkpoint_step: int) -> None:
    """Load the checkpoint."""
    persister = _get_persister(persisters)
    if persister:
        persister.load_weights(model, load_checkpoint_step)
        SIMPLEXITY_LOGGER.info("[predictive model] loaded checkpoint step: %s", load_checkpoint_step)
    else:
        raise RuntimeError("Unable to load model checkpoint")


def _setup_predictive_models(
    cfg: DictConfig, instance_keys: list[str], persisters: dict[str, ModelPersister] | None
) -> dict[str, Any] | None:
    """Setup the predictive model."""
    models = {}
    model_instance_keys = filter_instance_keys(cfg, instance_keys, is_predictive_model_target)
    for instance_key in model_instance_keys:
        instance_config: DictConfig | None = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
        if instance_config and is_hooked_transformer_config(instance_config):
            instance_config_config: DictConfig | None = instance_config.get("cfg", None)
            if instance_config_config is None:
                raise RuntimeError("Error selecting predictive model config")
            vocab_size = _get_attribute_value(cfg, instance_keys, "vocab_size")
            bos_token = _get_attribute_value(cfg, instance_keys, "bos_token")
            eos_token = _get_attribute_value(cfg, instance_keys, "eos_token")
            sequence_len = _get_attribute_value(cfg, instance_keys, "sequence_len")
            resolve_hooked_transformer_config(
                instance_config_config,
                vocab_size=vocab_size,
                bos_token=bos_token,
                eos_token=eos_token,
                sequence_len=sequence_len,
            )
        model = _instantiate_predictive_model(cfg, instance_key)
        step_key = instance_key.rsplit(".", 1)[0] + ".load_checkpoint_step"
        load_checkpoint_step: int | None = OmegaConf.select(cfg, step_key, throw_on_missing=True)
        if load_checkpoint_step is not None:
            _load_checkpoint(model, persisters, load_checkpoint_step)
        models[instance_key] = model
    if models:
        return models
    SIMPLEXITY_LOGGER.info("[predictive model] no predictive model config found")
    return None


def _get_predictive_model(predictive_models: dict[str, Any] | None) -> Any | None:
    if predictive_models:
        if len(predictive_models) == 1:
            return next(iter(predictive_models.values()))
        SIMPLEXITY_LOGGER.warning("Multiple predictive models found, any model checkpoint loading will be skipped")
        return None
    SIMPLEXITY_LOGGER.warning("No predictive model found, any model checkpoint loading will be skipped")
    return None


def _instantiate_optimizer(cfg: DictConfig, instance_key: str, predictive_model: Any | None) -> Any:
    """Setup the optimizer."""
    instance_config = OmegaConf.select(cfg, instance_key, throw_on_missing=True)
    if instance_config:
        if is_pytorch_optimizer_config(instance_config):
            if predictive_model and isinstance(predictive_model, PytorchModel):
                optimizer = hydra.utils.instantiate(instance_config, params=predictive_model.parameters())
                SIMPLEXITY_LOGGER.info("[optimizer] instantiated optimizer: %s", optimizer.__class__.__name__)
                return optimizer
            SIMPLEXITY_LOGGER.warning("Predictive model has no parameters, optimizer will be skipped")
            return None
        optimizer = hydra.utils.instantiate(instance_config)  # TODO: typed instantiate
        SIMPLEXITY_LOGGER.info("[optimizer] instantiated optimizer: %s", optimizer.__class__.__name__)
        return optimizer
    raise KeyError


def _setup_optimizers(
    cfg: DictConfig, instance_keys: list[str], predictive_models: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Setup the optimizer."""
    instance_keys = filter_instance_keys(
        cfg,
        instance_keys,
        is_optimizer_target,
        validate_fn=validate_optimizer_config,
        component_name="optimizer",
    )
    if instance_keys:
        model = _get_predictive_model(predictive_models)
        return {instance_key: _instantiate_optimizer(cfg, instance_key, model) for instance_key in instance_keys}
    SIMPLEXITY_LOGGER.info("[optimizer] no optimizer configs found")
    return None


def _do_logging(cfg: DictConfig, loggers: dict[str, Logger] | None, verbose: bool) -> None:
    if loggers is None:
        return
    for logger in loggers.values():
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
    _do_logging(cfg, components.loggers, verbose)
    return components


def _cleanup(components: Components) -> None:
    """Cleanup the run."""
    if components.loggers:
        for logger in components.loggers.values():
            logger.close()
    if components.persisters:
        for persister in components.persisters.values():
            persister.cleanup()


def managed_run(strict: bool = True, verbose: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Manage a run."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                cfg = get_config(args, kwargs)
                validate_base_config(cfg)
                with _setup_mlflow(cfg):
                    components = _setup(cfg, strict=strict, verbose=verbose)
                    output = fn(*args, **kwargs, components=components)
                _cleanup(components)
                return output
            except Exception as e:
                SIMPLEXITY_LOGGER.error("[run] error: %s", e)
                # TODO: cleanup
                raise e

        return wrapper

    return decorator
