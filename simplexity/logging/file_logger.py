from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from simplexity.logging.logger import Logger


class FileLogger(Logger):
    """Logs to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise RuntimeError(f"Failed to create directory for logging: {e}") from e

    def log_config(self, config: DictConfig) -> None:
        """Log config to the file."""
        with open(self.file_path, "a") as f:
            print(f"Config: {config}", file=f)

    def log_resolved_config(self, config: DictConfig) -> None:
        """Log a resolved config to the file."""
        with open(self.file_path, "a") as f:
            resolved_config = OmegaConf.to_container(config, resolve=True)
            print(f"Resolved config: {resolved_config}", file=f)

    def log_metrics(self, step: int, metric_dict: Mapping[str, Any]) -> None:
        """Log metrics to the file."""
        with open(self.file_path, "a") as f:
            print(f"Metrics at step {step}: {metric_dict}", file=f)

    def log_params(self, param_dict: Mapping[str, Any]) -> None:
        """Log params to the file."""
        with open(self.file_path, "a") as f:
            print(f"Params: {param_dict}", file=f)

    def log_tags(self, tag_dict: Mapping[str, Any]) -> None:
        """Log tags to the file."""
        with open(self.file_path, "a") as f:
            print(f"Tags: {tag_dict}", file=f)

    def close(self) -> None:
        """Close the logger."""
        pass


if __name__ == "__main__":
    print("Testing FileLogger resolved config...")
    logger = FileLogger("test.log")
    print(f"Logging to {logger.file_path}")
    logger.log_resolved_config(
        DictConfig(
            {
                "base_value": "hello",
                "interpolated_value": "${base_value}_world",
                "nested": {"value": "${base_value}_nested"},
            }
        )
    )
    logger.close()
    print("Test completed! Check test.log for output.")
