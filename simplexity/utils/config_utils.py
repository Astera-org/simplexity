from collections.abc import Callable
from typing import Any, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

T = TypeVar("T")
TARGET: str = "_target_"


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


def filter_instance_keys(cfg: DictConfig, instance_keys: list[str], filter_fn: Callable[[str], bool]) -> list[str]:
    """Filter instance keys by filter function to their targets."""
    filtered_instance_keys: list[str] = []
    for instance_key in instance_keys:
        target = OmegaConf.select(cfg, f"{instance_key}._target_", throw_on_missing=False)
        if isinstance(target, str) and filter_fn(target):
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


def typed_instantiate(config: Any, expected_type: type[T], **kwargs) -> T:
    """Instantiate an object from config with proper typing."""
    obj = hydra.utils.instantiate(config, **kwargs)
    assert isinstance(obj, expected_type)
    return obj
