from dataclasses import dataclass
from typing import Any


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str

    def __init__(self, _target_: str, **kwargs: Any):
        self._target_ = _target_
        for key, value in kwargs.items():
            setattr(self, key, value)
