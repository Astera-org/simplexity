from collections.abc import Callable
from typing import Any, Literal, TypeVar

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

T = TypeVar("T")
TARGET = Literal["_target_"]


def get_targets(cfg: DictConfig, *, nested: bool = False) -> list[str]:
    """Get targets."""
    targets: list[str] = []
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            if TARGET in cfg:
                targets.append(str(k))
            if TARGET not in cfg or nested:
                targets.extend([f"{k}.{target}" for target in get_targets(v, nested=nested)])

    return targets


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
