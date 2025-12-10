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

from unittest.mock import call, patch

import pytest
from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.base import resolve_base_config, validate_base_config


class TestValidateBaseConfig:
    """Test validate_base_config."""

    def test_validate_base_config_valid(self) -> None:
        """Test validate_base_config with valid configs."""
        cfg = DictConfig({})
        validate_base_config(cfg)

        cfg = DictConfig(
            {
                "device": "auto",
                "seed": 42,
                "tags": DictConfig({"key": "value"}),
                "mlflow": DictConfig({"experiment_name": "test", "run_name": "test"}),
            }
        )
        validate_base_config(cfg)

    def test_validate_base_config_invalid_device(self) -> None:
        """Test validate_base_config with invalid device."""
        cfg = DictConfig({"device": ""})
        with pytest.raises(ConfigValidationError, match="BaseConfig.device must be a non-empty string"):
            validate_base_config(cfg)

        cfg = DictConfig({"device": 0})
        with pytest.raises(ConfigValidationError, match="BaseConfig.device must be a string"):
            validate_base_config(cfg)

        cfg = DictConfig({"device": "invalid"})
        with pytest.raises(
            ConfigValidationError, match="BaseConfig.device must be one of: \\('auto', 'cpu', 'gpu', 'cuda'\\)"
        ):
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


class TestResolveBaseConfig:
    """Test resolve_base_config."""

    def test_empty_config_with_explicit_param_values(self) -> None:
        """Test resolve_base_config with valid configs."""
        cfg = DictConfig({})
        resolve_base_config(cfg, strict=True, seed=34, device="gpu")
        assert cfg.device == "gpu"
        assert cfg.seed == 34
        assert cfg.tags.strict == "true"

    def test_empty_config_with_default_param_values(self) -> None:
        """Test resolve_base_config with default values."""
        cfg = DictConfig({})
        resolve_base_config(cfg, strict=False)
        assert cfg.device == "auto"
        assert cfg.seed == 42
        assert cfg.tags.strict == "false"

    def test_config_with_matching_param_values(self) -> None:
        """Test resolve_base_config overrides mismatched seed and strict values."""
        # matching values
        cfg = DictConfig({"device": "gpu", "seed": 34, "tags": DictConfig({"strict": "true"})})
        resolve_base_config(cfg, strict=True, seed=34, device="gpu")
        assert cfg.device == "gpu"
        assert cfg.seed == 34
        assert cfg.tags.strict == "true"

    def test_config_with_non_matching_param_values(self) -> None:
        """Test resolve_base_config overrides mismatched seed and strict values."""
        # non-matching values
        cfg = DictConfig({"device": "gpu", "seed": 34, "tags": DictConfig({"strict": "true"})})
        with patch("simplexity.structured_configs.base.SIMPLEXITY_LOGGER.warning") as mock_warning:
            resolve_base_config(cfg, strict=False, seed=56, device="cpu")
            mock_warning.assert_has_calls(
                [
                    call("Device tag set to '%s', but device is '%s'. Overriding device tag.", "gpu", "cpu"),
                    call("Seed tag set to '%s', but seed is '%s'. Overriding seed tag.", 34, 56),
                    call("Strict tag set to '%s', but strict mode is '%s'. Overriding strict tag.", "true", "false"),
                ]
            )
            assert cfg.device == "cpu"
            assert cfg.seed == 56
            assert cfg.tags.strict == "false"

    def test_config_with_no_param_values(self) -> None:
        """Test resolve_base_config with no param values."""
        cfg = DictConfig({"device": "gpu", "seed": 34})
        resolve_base_config(cfg, strict=False)
        assert cfg.device == "gpu"
        assert cfg.seed == 34
        assert cfg.tags.strict == "false"
