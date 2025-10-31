from collections.abc import Callable
from typing import Any, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

T = TypeVar("T")
TARGET: str = "_target_"


def get_targets(cfg: DictConfig, *, nested: bool = False) -> list[str]:
    """Get targets."""
    targets: list[str] = []
    for key in cfg:
        try:
            value = cfg[key]
        except MissingMandatoryValue:
            continue
        if isinstance(value, DictConfig):
            if TARGET in value:
                targets.append(str(key))
            if TARGET not in value or nested:
                targets.extend([f"{key}.{target}" for target in get_targets(value, nested=nested)])

    return targets


def filter_targets(cfg: DictConfig, targets: list[str], prefix: str) -> list[str]:
    """Filter targets by prefix."""
    filtered_targets: list[str] = []
    for target in targets:
        target_value = OmegaConf.select(cfg, f"{target}._target_", throw_on_missing=False)
        if isinstance(target_value, str) and target_value.startswith(prefix):
            filtered_targets.append(target)
    return filtered_targets


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
