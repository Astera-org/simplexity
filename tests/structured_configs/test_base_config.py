"""Tests for BaseConfig validation.

This module contains tests for the base configuration validation functionality,
including validation of seed, tags, and MLFlow configuration fields.
"""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import pytest
from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.base import validate_base_config


class TestBaseConfig:
    """Test BaseConfig."""

    def test_validate_base_config_valid(self) -> None:
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

    def test_validate_base_config_invalid_seed(self) -> None:
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

    def test_validate_base_config_invalid_tags(self) -> None:
        """Test validate_base_config with invalid tags."""
        # Non-dictionary tags
        cfg = DictConfig({"tags": "not a dictionary"})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags must be a dictionary"):
            validate_base_config(cfg)

        # Tags with non-string keys
        cfg = DictConfig({"tags": {123: "value"}})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags keys must be strs"):
            validate_base_config(cfg)

        # Tags with non-string values
        cfg = DictConfig({"tags": {"key": 123}})
        with pytest.raises(ConfigValidationError, match="BaseConfig.tags values must be strs"):
            validate_base_config(cfg)

    def test_validate_base_config_invalid_mlflow(self) -> None:
        """Test validate_base_config with invalid mlflow."""
        # Non-MLFlowConfig mlflow
        cfg = DictConfig({"mlflow": "not an MLFlowConfig"})
        with pytest.raises(ConfigValidationError, match="BaseConfig.mlflow must be a MLFlowConfig"):
            validate_base_config(cfg)

        # MLFlowConfig with empty experiment_name (whitespace)
        cfg = DictConfig({"mlflow": DictConfig({"experiment_name": "  "})})
        with pytest.raises(ConfigValidationError, match="MLFlowConfig.experiment_name must be a non-empty string"):
            validate_base_config(cfg)

    def test_validate_base_config_propagates_mlflow_errors(self) -> None:
        """Test that MLflow validation errors propagate correctly."""
        # Invalid tracking_uri scheme
        cfg = DictConfig({"mlflow": DictConfig({"tracking_uri": "relative/path"})})
        with pytest.raises(ConfigValidationError, match="MLFlowConfig.tracking_uri must have a valid URI scheme"):
            validate_base_config(cfg)

        # Empty experiment_name
        cfg = DictConfig({"mlflow": DictConfig({"experiment_name": "  "})})
        with pytest.raises(ConfigValidationError, match="MLFlowConfig.experiment_name must be a non-empty string"):
            validate_base_config(cfg)
