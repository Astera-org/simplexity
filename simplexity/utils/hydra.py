from typing import Any, Literal, TypeVar

import hydra
from omegaconf import DictConfig

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


def typed_instantiate(config: Any, expected_type: type[T], **kwargs) -> T:
    """Instantiate an object from config with proper typing."""
    obj = hydra.utils.instantiate(config, **kwargs)
    assert isinstance(obj, expected_type)
    return obj
