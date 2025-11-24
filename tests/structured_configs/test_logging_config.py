"""Tests for LoggingConfig validation.

This module contains tests for logging configuration validation, including
validation of logger targets, logger configs, and logging configuration instances.
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
from omegaconf import DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.logging import (
    FileLoggerInstanceConfig,
    InstanceConfig,
    LoggingConfig,
    is_logger_config,
    is_logger_target,
    update_logging_instance_config,
    validate_logging_config,
)


class TestLoggingConfig:
    """Test LoggingConfig."""

    def test_logging_config(self) -> None:
        """Test creating logger config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(LoggingConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_logger_target_valid(self) -> None:
        """Test is_logger_target with valid logger targets."""
        assert is_logger_target("simplexity.logging.file_logger.FileLogger")
        assert is_logger_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert is_logger_target("simplexity.logging.print_logger.PrintLogger")

    def test_is_logger_target_invalid(self) -> None:
        """Test is_logger_target with invalid targets."""
        assert not is_logger_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert not is_logger_target("logging.Logger")
        assert not is_logger_target("")

    def test_is_logger_config_valid(self) -> None:
        """Test is_logger_config with valid logger configs."""
        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert is_logger_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                "tracking_uri": "databricks",
            }
        )
        assert is_logger_config(cfg)

    def test_is_logger_config_invalid(self) -> None:
        """Test is_logger_config with invalid configs."""
        # Non-logger target
        cfg = DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})
        assert not is_logger_config(cfg)

        # Missing _target_
        cfg = DictConfig({"experiment_name": "my_experiment", "run_name": "my_run", "tracking_uri": "databricks"})
        assert not is_logger_config(cfg)

        # _target_ is not a omegaconf target
        cfg = DictConfig({"target": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_logger_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_logger_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_logger_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_logger_config(cfg)

    def test_validate_logging_config_valid(self) -> None:
        """Test validate_logging_config with valid configs."""
        # Valid config without name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                        "tracking_uri": "databricks",
                    }
                ),
            }
        )
        validate_logging_config(cfg)  # Should not raise

        # Valid config with name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                        "tracking_uri": "databricks",
                    }
                ),
                "name": "my_logger",
            }
        )
        validate_logging_config(cfg)  # Should not raise

        # Valid config with None name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                        "tracking_uri": "databricks",
                    }
                ),
                "name": None,
            }
        )
        validate_logging_config(cfg)  # Should not raise

    def test_validate_logging_config_missing_instance(self) -> None:
        """Test validate_logging_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance must be a DictConfig"):
            validate_logging_config(cfg)

        cfg = DictConfig({"name": "my_logger"})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance must be a DictConfig"):
            validate_logging_config(cfg)

    def test_validate_logging_config_invalid_instance(self) -> None:
        """Test validate_logging_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_logging_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_logging_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_logging_config(cfg)

    def test_validate_logging_config_non_logger_target(self) -> None:
        """Test validate_logging_config raises when instance target is not a logger target."""
        cfg = DictConfig(
            {"instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})}
        )
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance must be a logger target"):
            validate_logging_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance must be a logger target"):
            validate_logging_config(cfg)

    def test_validate_logging_config_invalid_name(self) -> None:
        """Test validate_logging_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be a non-empty string"):
            validate_logging_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be a non-empty string"):
            validate_logging_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be a string or None"):
            validate_logging_config(cfg)

    def test_validate_file_logger_config(self) -> None:
        """Test validation of FileLogger configuration."""
        # Valid file logger config
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.file_logger.FileLogger",
                        "file_path": "/tmp/test.log",
                    }
                )
            }
        )
        validate_logging_config(cfg)

        # Missing file_path
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.file_logger.FileLogger",
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="FileLoggerInstanceConfig.file_path must be a string"):
            validate_logging_config(cfg)

        # Empty file_path
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.logging.file_logger.FileLogger",
                        "file_path": "",
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="FileLoggerInstanceConfig.file_path must be a non-empty string"):
            validate_logging_config(cfg)

    def test_update_logging_instance_config(self) -> None:
        """Test update_logging_instance_config function."""
        # Initial config
        cfg = DictConfig(
            {
                "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                "experiment_name": "exp1",
                "run_name": "run1",
            }
        )

        # Update config
        updated_cfg = DictConfig(
            {
                "_target_": "simplexity.logging.mlflow_logger.MLFlowLogger",
                "experiment_name": "exp2",
                "tracking_uri": "file:///tmp/mlruns",
            }
        )

        update_logging_instance_config(cfg, updated_cfg)

        assert cfg.experiment_name == "exp2"
        assert cfg.run_name == "run1"  # Should remain unchanged
        assert cfg.tracking_uri == "file:///tmp/mlruns"

    def test_file_logger_instance_config_init(self) -> None:
        """Test FileLoggerInstanceConfig instantiation."""
        config = FileLoggerInstanceConfig(file_path="test.log")
        assert config.file_path == "test.log"
        assert config._target_ == "simplexity.logging.file_logger.FileLogger"
