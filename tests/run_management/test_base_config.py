"""Tests for BaseConfig validation.

This module contains tests for the base configuration validation functionality,
including validation of seed, tags, and MLFlow configuration fields.
"""

import pytest
from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.run_management.structured_configs import validate_base_config


class TestBaseConfig:
    def test_validate_base_config_valid(self):
        """Test validate_base_config with valid configs."""
        cfg = DictConfig({})
        validate_base_config(cfg)

        cfg = DictConfig(
            {
                "seed": 42,
                "tags": DictConfig({"key": "value"}),
                "mlflow": DictConfig({"experiment_name": "test", "run_name": "test"}),
            }
        )
        validate_base_config(cfg)

    def test_validate_base_config_invalid_seed(self):
        """Test validate_base_config with invalid configs."""
        # Non-integer seed
        cfg = DictConfig({"seed": "42"})
        with pytest.raises(ConfigValidationError, match="BaseConfig.seed must be an int or None"):
            validate_base_config(cfg)

        cfg = DictConfig({"seed": False})
        with pytest.raises(ConfigValidationError, match="BaseConfig.seed must be an int or None"):
            validate_base_config(cfg)

        # Negative seed
        cfg = DictConfig({"seed": -1})
        with pytest.raises(ConfigValidationError, match="BaseConfig.seed must be non-negative"):
            validate_base_config(cfg)

    def test_validate_base_config_invalid_tags(self):
        """Test validate_base_config with invalid tags."""
        # Non-dictionary tags
        cfg = DictConfig({"tags": "not a dictionary"})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags must be a dictionary"):
            validate_base_config(cfg)

        # Tags with non-string keys
        cfg = DictConfig({"tags": {123: "value"}})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags keys must be strings"):
            validate_base_config(cfg)

        # Tags with non-string values
        cfg = DictConfig({"tags": {"key": 123}})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags values must be strings"):
            validate_base_config(cfg)

    def test_validate_base_config_invalid_mlflow(self):
        """Test validate_base_config with invalid mlflow."""
        # Non-MLFlowConfig mlflow
        cfg = DictConfig({"mlflow": "not an MLFlowConfig"})
        with pytest.raises(ConfigValidationError, match="BaseConfig.mlflow must be a MLFlowConfig"):
            validate_base_config(cfg)

        # MLFlowConfig with missing experiment_name
        cfg = DictConfig({"mlflow": DictConfig({"run_name": "test"})})
        with pytest.raises(ConfigValidationError, match="MLFlowConfig.experiment_name must be a non-empty string"):
            validate_base_config(cfg)
