from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig


class Logger(ABC):
    """Logs to a variety of backends."""

    @abstractmethod
    def log_config(self, config: DictConfig) -> None:
        """Log config to the logger."""
        ...

    @abstractmethod
    def log_resolved_config(self, config: DictConfig) -> None:
        """Log a resolved config to the logger."""
        ...

    @abstractmethod
    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the logger."""
        ...

    @abstractmethod
    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the logger."""
        ...

    @abstractmethod
    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the logger."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        ...
