"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import validate_mapping, validate_nonempty_str, validate_sequence


@dataclass
class MetricTrackerInstanceConfig(InstanceConfig):
    """Configuration for MetricTracker instance."""

    metric_names: dict[str, list[str]] | list[str] | None = None
    metric_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        metric_names: dict[str, list[str]] | list[str] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        _target_: str = "simplexity.metrics.metric_tracker.MetricTracker",
    ):
        super().__init__(_target_=_target_)
        self.metric_names = metric_names
        self.metric_kwargs = metric_kwargs


@dataclass
class MetricTrackerConfig:
    """Base configuration for metric trackers."""

    instance: MetricTrackerInstanceConfig | InstanceConfig
    name: str | None = None


def is_metric_tracker_target(target: str) -> bool:
    """Check if the target is a metric tracker target."""
    return target == "simplexity.metrics.metric_tracker.MetricTracker"


def is_metric_tracker_config(cfg: DictConfig) -> bool:
    """Check if the configuration is a metric tracker config."""
    target = cfg.get("_target_", None)
    if isinstance(target, str):
        return is_metric_tracker_target(target)
    return False


def validate_metric_tracker_instance_config(cfg: DictConfig) -> None:
    """Validate a MetricTrackerInstanceConfig.

    Args:
        cfg: A DictConfig with _target_, metric_names, and metric_kwargs fields (from Hydra).
    """
    validate_instance_config(cfg, expected_target="simplexity.metrics.metric_tracker.MetricTracker")

    metric_names = cfg.get("metric_names")
    if metric_names is not None:
        if isinstance(metric_names, DictConfig):
            validate_mapping(metric_names, "MetricTrackerInstanceConfig.metric_names", key_type=str)
            for value in metric_names.values():
                validate_sequence(value, "MetricTrackerInstanceConfig.metric_names", element_type=str)
        elif isinstance(metric_names, list):
            validate_sequence(metric_names, "MetricTrackerInstanceConfig.metric_names", element_type=str)
        else:
            raise ConfigValidationError("MetricTrackerInstanceConfig.metric_names must be a dict or list")

    metric_kwargs = cfg.get("metric_kwargs")
    validate_mapping(metric_kwargs, "MetricTrackerInstanceConfig.metric_kwargs", key_type=str, is_none_allowed=True)


def validate_metric_tracker_config(cfg: DictConfig) -> None:
    """Validate a MetricTrackerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("MetricTrackerConfig.instance is required")
    validate_metric_tracker_instance_config(instance)

    validate_nonempty_str(cfg.get("name"), "MetricTrackerConfig.name", is_none_allowed=True)
