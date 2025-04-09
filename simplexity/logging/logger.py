from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
import yaml

from omegaconf import OmegaConf, DictConfig


class Logger(ABC):
    """Logs to a variety of backends."""

    @abstractmethod
    def log_config(self, config: DictConfig) -> None:
        """Log config to the logger."""
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

def config_to_yaml_string(cfg: DictConfig):
    # Convert OmegaConf to dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Filter out _target_ and _self_ fields recursively
    def filter_special_fields(d):
        if isinstance(d, dict):
            return {k: filter_special_fields(v) for k, v in d.items() 
                   if k != '_target_' and k != '_self_'}
        elif isinstance(d, list):
            return [filter_special_fields(i) for i in d]
        else:
            return d
    
    filtered_dict = filter_special_fields(cfg_dict)
    
    # Convert to YAML string
    yaml_str = yaml.dump(filtered_dict, default_flow_style=False, sort_keys=False)
    return yaml_str

