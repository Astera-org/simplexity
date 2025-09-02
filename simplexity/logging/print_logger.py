from collections.abc import Mapping
from pprint import pprint
from typing import Any

from omegaconf import DictConfig, OmegaConf

from simplexity.logging.logger import Logger


class PrintLogger(Logger):
    """Logs to the console."""

    def log_config(self, config: DictConfig) -> None:
        """Log config to the console."""
        pprint(f"Config: {config}")

    def log_resolved_config(self, config: DictConfig) -> None:
        """Log a resolved config to the console."""
        resolved_config = OmegaConf.to_container(config, resolve=True)
        pprint(f"Resolved config: {resolved_config}")

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the console."""
        pprint(f"Metrics at step {step}: {metric_dict}")

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the console."""
        pprint(f"Params: {param_dict}")

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the console."""
        pprint(f"Tags: {tag_dict}")

    def close(self) -> None:
        """Close the logger."""
        pass
