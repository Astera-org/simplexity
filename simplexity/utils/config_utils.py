"""Config utilities."""

import importlib
import logging
from collections.abc import Callable
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

from simplexity.exceptions import ConfigValidationError

TARGET: str = "_target_"
SIMPLEXITY_LOGGER = logging.getLogger("simplexity")


def get_instance_keys(cfg: DictConfig, *, nested: bool = False) -> list[str]:
    """Get instance keys."""
    instance_keys: list[str] = []
    for key in cfg:
        try:
            value = cfg[key]
        except MissingMandatoryValue:
            continue
        if isinstance(value, DictConfig):
            if TARGET in value:
                instance_keys.append(str(key))
            if TARGET not in value or nested:
                instance_keys.extend([f"{key}.{target}" for target in get_instance_keys(value, nested=nested)])

    return instance_keys


def _validate(
    cfg: DictConfig,
    instance_key: str,
    validate_fn: Callable[[DictConfig], None] | None,
    component_name: str | None = None,
) -> bool:
    if validate_fn is None:
        return True
    config_key = instance_key.rsplit(".", 1)[0]
    config: DictConfig | None = OmegaConf.select(cfg, config_key, throw_on_missing=True)
    if config is None:
        return False
    try:
        validate_fn(config)
    except ConfigValidationError as e:
        component_prefix = f"[{component_name}] " if component_name else ""
        SIMPLEXITY_LOGGER.warning("%serror validating config: %s", component_prefix, e)
        return False
    return True


def filter_instance_keys(
    cfg: DictConfig,
    instance_keys: list[str],
    filter_fn: Callable[[str], bool],
    validate_fn: Callable[[DictConfig], None] | None = None,
    component_name: str | None = None,
) -> list[str]:
    """Filter instance keys by filter function to their targets."""
    filtered_instance_keys: list[str] = []
    for instance_key in instance_keys:
        target = OmegaConf.select(cfg, f"{instance_key}._target_", throw_on_missing=False)
        if isinstance(target, str) and filter_fn(target) and _validate(cfg, instance_key, validate_fn, component_name):
            filtered_instance_keys.append(instance_key)
    return filtered_instance_keys


def get_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DictConfig:
    """Get the config from the arguments."""
    if kwargs and "cfg" in kwargs:
        return kwargs["cfg"]
    if args and isinstance(args[0], DictConfig):
        return args[0]
    raise ValueError("No config found in arguments or kwargs.")


def dynamic_resolve(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Dynamic resolve decorator."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cfg = get_config(args, kwargs)
        with open_dict(cfg):
            output = fn(*args, **kwargs)
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, True)
        OmegaConf.set_readonly(cfg, True)
        return output

    return wrapper


def _resolve_target(target_str: str) -> type:
    module_path, _, cls_name = target_str.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def typed_instantiate[T](config: Any, expected_type: type[T] | str, **kwargs) -> T:
    """Instantiate an object from config with proper typing."""
    if isinstance(expected_type, str):
        expected_type = _resolve_target(expected_type)
    obj = hydra.utils.instantiate(config, **kwargs)
    assert isinstance(obj, expected_type)
    return obj
