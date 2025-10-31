from dataclasses import dataclass


@dataclass
class InstanceConfig:
    """Config for an object that can be instantiated by hydra."""

    _target_: str
