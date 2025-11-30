"""Structured configuration dataclasses for all components.

This module centralizes all structured config definitions that were previously
scattered across various config.py files in the configs directory.
"""

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config
from simplexity.structured_configs.validation import validate_nonempty_str


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
    return target.startswith("simplexity.metrics.")


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
    validate_instance_config(cfg)

    # Validate metric_names if provided
    metric_names = cfg.get("metric_names")
    if metric_names is not None:
        if isinstance(metric_names, DictConfig):
            # Validate dict format: {group_name: [metric1, metric2, ...]}
            for key, value in metric_names.items():
                if not isinstance(key, str):
                    raise ConfigValidationError(
                        f"MetricTrackerInstanceConfig.metric_names keys must be strings, got {type(key)}"
                    )
                if not isinstance(value, (list, DictConfig)):
                    raise ConfigValidationError(
                        f"MetricTrackerInstanceConfig.metric_names['{key}'] must be a list, got {type(value)}"
                    )
                if isinstance(value, list):
                    for item in value:
                        if not isinstance(item, str):
                            raise ConfigValidationError(
                                f"MetricTrackerInstanceConfig.metric_names['{key}'] items must be strings, "
                                f"got {type(item)}"
                            )
        elif isinstance(metric_names, (list, DictConfig)):
            # Validate list format: [metric1, metric2, ...]
            if isinstance(metric_names, list):
                for item in metric_names:
                    if not isinstance(item, str):
                        raise ConfigValidationError(
                            f"MetricTrackerInstanceConfig.metric_names items must be strings, got {type(item)}"
                        )
        else:
            raise ConfigValidationError(
                f"MetricTrackerInstanceConfig.metric_names must be a dict or list, got {type(metric_names)}"
            )

    # Validate metric_kwargs if provided
    metric_kwargs = cfg.get("metric_kwargs")
    if metric_kwargs is not None and not isinstance(metric_kwargs, DictConfig):
        raise ConfigValidationError(
            f"MetricTrackerInstanceConfig.metric_kwargs must be a dict, got {type(metric_kwargs)}"
        )


def validate_metric_tracker_config(cfg: DictConfig) -> None:
    """Validate a MetricTrackerConfig.

    Args:
        cfg: A DictConfig with instance and optional name fields (from Hydra).
    """
    instance = cfg.get("instance")
    if instance is None:
        raise ConfigValidationError("MetricTrackerConfig.instance is required")

    target = instance.get("_target_", None)
    if not is_metric_tracker_target(target):
        raise ConfigValidationError(
            f"MetricTrackerConfig.instance._target_ must be a metric tracker target, got {target}"
        )

    # Validate the instance config
    validate_metric_tracker_instance_config(instance)

    validate_nonempty_str(cfg.get("name"), "MetricTrackerConfig.name", is_none_allowed=True)
