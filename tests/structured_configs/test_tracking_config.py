"""Tests for TrackingConfig validation.

This module contains tests for tracking configuration validation, including
validation of tracker targets, tracker configs, and tracking configuration instances.
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
from simplexity.structured_configs.tracking import (
    FileTrackerInstanceConfig,
    InstanceConfig,
    MlflowTrackerInstanceConfig,
    S3TrackerInstanceConfig,
    TrackingConfig,
    is_file_tracker_config,
    is_file_tracker_target,
    is_mlflow_tracker_config,
    is_mlflow_tracker_target,
    is_run_tracker_config,
    is_run_tracker_target,
    is_s3_tracker_config,
    is_s3_tracker_target,
    update_tracking_instance_config,
    validate_file_tracker_instance_config,
    validate_mlflow_tracker_instance_config,
    validate_s3_tracker_instance_config,
    validate_tracking_config,
)


class TestTrackingConfig:
    """Test TrackingConfig."""

    def test_structured_config(self) -> None:
        """Test creating tracking config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(TrackingConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_structured_config_with_name(self) -> None:
        """Test creating tracking config with name."""
        cfg: DictConfig = OmegaConf.structured(
            TrackingConfig(instance=InstanceConfig(_target_="some_target"), name="my_tracker")
        )
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") == "my_tracker"

    def test_is_tracker_target(self) -> None:
        """Test is_tracker_target with valid targets."""
        assert is_run_tracker_target("simplexity.tracking.file_tracker.FileTracker")
        assert is_run_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker")
        assert is_run_tracker_target("simplexity.tracking.s3_tracker.S3Tracker.from_config")

    def test_is_tracker_target_invalid(self) -> None:
        """Test is_tracker_target with invalid targets."""
        assert not is_run_tracker_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_run_tracker_target("some.other.tracker.Tracker")
        assert not is_run_tracker_target("")

    def test_validate_tracking_config_valid(self) -> None:
        """Test validate_tracking_config with valid configs."""
        # Valid config without name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                    }
                ),
            }
        )
        validate_tracking_config(cfg)

        # Valid config with name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                    }
                ),
                "name": "my_tracker",
            }
        )
        validate_tracking_config(cfg)

        # Valid config with None name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                    }
                ),
                "name": None,
            }
        )
        validate_tracking_config(cfg)

    def test_validate_tracking_config_missing_instance(self) -> None:
        """Test validate_tracking_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="TrackingConfig.instance must be a DictConfig"):
            validate_tracking_config(cfg)

        cfg = DictConfig({"name": "my_tracker"})
        with pytest.raises(ConfigValidationError, match="TrackingConfig.instance must be a DictConfig"):
            validate_tracking_config(cfg)

    def test_validate_tracking_config_invalid_instance(self) -> None:
        """Test validate_tracking_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_tracking_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_tracking_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_tracking_config(cfg)

    def test_validate_tracking_config_non_tracker_target(self) -> None:
        """Test validate_tracking_config raises when instance target is not a tracker target."""
        cfg = DictConfig(
            {"instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})}
        )
        with pytest.raises(ConfigValidationError, match="TrackingConfig.instance must be a tracker target"):
            validate_tracking_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(ConfigValidationError, match="TrackingConfig.instance must be a tracker target"):
            validate_tracking_config(cfg)

    def test_validate_tracking_config_invalid_name(self) -> None:
        """Test validate_tracking_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker"}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="TrackingConfig.name must be a non-empty string"):
            validate_tracking_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker"}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="TrackingConfig.name must be a non-empty string"):
            validate_tracking_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker"}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="TrackingConfig.name must be a string or None"):
            validate_tracking_config(cfg)


class TestFileTrackerConfig:
    """Test FileTracker configuration functions."""

    def test_is_file_tracker_target_valid(self) -> None:
        """Test is_file_tracker_target with valid target."""
        assert is_file_tracker_target("simplexity.tracking.file_tracker.FileTracker")

    def test_is_file_tracker_target_invalid(self) -> None:
        """Test is_file_tracker_target with invalid targets."""
        assert not is_file_tracker_target("simplexity.tracking.file_tracker.FileTracker.from_config")
        assert not is_file_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker")

    def test_is_file_tracker_config_valid(self) -> None:
        """Test is_file_tracker_config with valid configs."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "my_file.log",
            }
        )
        assert is_file_tracker_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "my_file.log",
                "model_dir_name": "custom_models",
            }
        )
        assert is_file_tracker_config(cfg)

    def test_is_file_tracker_config_invalid(self) -> None:
        """Test is_file_tracker_config with invalid configs."""
        # Non-file tracker target
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "file_path": "my_file.log",
            }
        )
        assert not is_file_tracker_config(cfg)

        # Missing _target_
        cfg = DictConfig({"file_path": "my_file.log"})
        assert not is_file_tracker_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_file_tracker_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_file_tracker_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_file_tracker_config(cfg)

    def test_validate_file_tracker_instance_config_valid(self) -> None:
        """Test validate_file_tracker_instance_config with valid configs."""
        # Valid config with required fields
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "/tmp/test.log",
            }
        )
        validate_file_tracker_instance_config(cfg)  # Should not raise

        # Valid config with optional model_dir_name
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "/tmp/test.log",
                "model_dir_name": "custom_models",
            }
        )
        validate_file_tracker_instance_config(cfg)  # Should not raise

    def test_validate_file_tracker_instance_config_invalid_target(self) -> None:
        """Test validate_file_tracker_instance_config raises with invalid target."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "file_path": "/tmp/test.log",
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be"):
            validate_file_tracker_instance_config(cfg)

    def test_validate_file_tracker_instance_config_missing_file_path(self) -> None:
        """Test validate_file_tracker_instance_config raises when file_path is missing."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
            }
        )
        with pytest.raises(ConfigValidationError, match="FileTrackerInstanceConfig.file_path must be a string"):
            validate_file_tracker_instance_config(cfg)

    def test_validate_file_tracker_instance_config_empty_file_path(self) -> None:
        """Test validate_file_tracker_instance_config raises when file_path is empty."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="FileTrackerInstanceConfig.file_path must be a non-empty string"
        ):
            validate_file_tracker_instance_config(cfg)

    def test_validate_file_tracker_instance_config_whitespace_file_path(self) -> None:
        """Test validate_file_tracker_instance_config raises when file_path is whitespace only."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "   ",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="FileTrackerInstanceConfig.file_path must be a non-empty string"
        ):
            validate_file_tracker_instance_config(cfg)

    def test_validate_file_tracker_instance_config_empty_model_dir_name(self) -> None:
        """Test validate_file_tracker_instance_config raises when model_dir_name is empty."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "/tmp/test.log",
                "model_dir_name": "",
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="FileTrackerInstanceConfig.model_dir_name must be a non-empty string",
        ):
            validate_file_tracker_instance_config(cfg)

    def test_file_tracker_instance_config_init(self) -> None:
        """Test FileTrackerInstanceConfig instantiation."""
        config = FileTrackerInstanceConfig(file_path="test.log")
        assert config.file_path == "test.log"
        assert config._target_ == "simplexity.tracking.file_tracker.FileTracker"
        assert config.model_dir_name == "models"  # Default value

    def test_file_tracker_instance_config_init_with_custom_model_dir(self) -> None:
        """Test FileTrackerInstanceConfig instantiation with custom model_dir_name."""
        config = FileTrackerInstanceConfig(file_path="test.log", model_dir_name="custom_models")
        assert config.file_path == "test.log"
        assert config.model_dir_name == "custom_models"
        assert config._target_ == "simplexity.tracking.file_tracker.FileTracker"

    def test_validate_tracking_config_with_file_tracker(self) -> None:
        """Test validate_tracking_config with FileTracker instance."""
        # Valid file tracker config
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.file_tracker.FileTracker",
                        "file_path": "/tmp/test.log",
                    }
                )
            }
        )
        validate_tracking_config(cfg)

        # Missing file_path
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.file_tracker.FileTracker",
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="FileTrackerInstanceConfig.file_path must be a string"):
            validate_tracking_config(cfg)

        # Empty file_path
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.file_tracker.FileTracker",
                        "file_path": "",
                    }
                )
            }
        )
        with pytest.raises(
            ConfigValidationError, match="FileTrackerInstanceConfig.file_path must be a non-empty string"
        ):
            validate_tracking_config(cfg)


class TestMlflowTrackerConfig:
    """Test MlflowTracker configuration functions."""

    def test_is_mlflow_tracker_target_valid(self) -> None:
        """Test is_mlflow_tracker_target with valid target."""
        assert is_mlflow_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker")

    def test_is_mlflow_tracker_target_invalid(self) -> None:
        """Test is_mlflow_tracker_target with invalid targets."""
        assert not is_mlflow_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker.from_config")
        assert not is_mlflow_tracker_target("simplexity.tracking.file_tracker.FileTracker")
        assert not is_mlflow_tracker_target("simplexity.tracking.s3_tracker.S3Tracker.from_config")
        assert not is_mlflow_tracker_target("")
        assert not is_mlflow_tracker_target("some.other.target")

    def test_is_mlflow_tracker_config_valid(self) -> None:
        """Test is_mlflow_tracker_config with valid configs."""
        cfg = DictConfig({"_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker"})
        assert is_mlflow_tracker_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_name": "my_experiment",
                "run_name": "my_run",
            }
        )
        assert is_mlflow_tracker_config(cfg)

    def test_is_mlflow_tracker_config_invalid(self) -> None:
        """Test is_mlflow_tracker_config with invalid configs."""
        # Non-mlflow tracker target
        cfg = DictConfig({"_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config"})
        assert not is_mlflow_tracker_config(cfg)

        # Missing _target_
        cfg = DictConfig({"experiment_name": "my_experiment", "run_name": "my_run"})
        assert not is_mlflow_tracker_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_mlflow_tracker_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_mlflow_tracker_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_mlflow_tracker_config(cfg)

    def test_validate_mlflow_tracker_instance_config_valid(self) -> None:
        """Test validate_mlflow_tracker_instance_config with valid configs."""
        # Valid config with minimal fields
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
            }
        )
        validate_mlflow_tracker_instance_config(cfg)  # Should not raise

        # Valid config with all optional fields
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_id": "exp123",
                "experiment_name": "my_experiment",
                "run_id": "run456",
                "run_name": "my_run",
                "tracking_uri": "databricks",
                "registry_uri": "databricks",
                "downgrade_unity_catalog": True,
                "model_dir": "models",
                "config_path": "config.yaml",
            }
        )
        validate_mlflow_tracker_instance_config(cfg)  # Should not raise

        # Valid config with None optional fields
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_id": None,
                "experiment_name": None,
                "run_id": None,
                "run_name": None,
                "tracking_uri": None,
                "registry_uri": None,
            }
        )
        validate_mlflow_tracker_instance_config(cfg)  # Should not raise

    def test_validate_mlflow_tracker_instance_config_invalid_target(self) -> None:
        """Test validate_mlflow_tracker_instance_config raises with invalid target."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "experiment_name": "my_experiment",
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be"):
            validate_mlflow_tracker_instance_config(cfg)

    def test_validate_mlflow_tracker_instance_config_empty_string_fields(self) -> None:
        """Test validate_mlflow_tracker_instance_config raises with empty string fields."""
        # Empty experiment_id
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_id": "",
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="MlflowTrackerInstanceConfig.experiment_id must be a non-empty string",
        ):
            validate_mlflow_tracker_instance_config(cfg)

        # Empty experiment_name
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_name": "",
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="MlflowTrackerInstanceConfig.experiment_name must be a non-empty string",
        ):
            validate_mlflow_tracker_instance_config(cfg)

        # Empty model_dir
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "model_dir": "",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="MlflowTrackerInstanceConfig.model_dir must be a non-empty string"
        ):
            validate_mlflow_tracker_instance_config(cfg)

    def test_validate_mlflow_tracker_instance_config_invalid_uri(self) -> None:
        """Test validate_mlflow_tracker_instance_config raises with invalid URIs."""
        # Invalid tracking_uri
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "tracking_uri": "  ",
            }
        )
        with pytest.raises(ConfigValidationError, match="MlflowTrackerInstanceConfig.tracking_uri"):
            validate_mlflow_tracker_instance_config(cfg)

        # Invalid registry_uri
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "registry_uri": "%parse_error%",
            }
        )
        with pytest.raises(ConfigValidationError, match="MlflowTrackerInstanceConfig.registry_uri"):
            validate_mlflow_tracker_instance_config(cfg)

    def test_mlflow_tracker_instance_config_init(self) -> None:
        """Test MlflowTrackerInstanceConfig instantiation."""
        config = MlflowTrackerInstanceConfig()
        assert config._target_ == "simplexity.tracking.mlflow_tracker.MlflowTracker"
        assert config.experiment_id is None
        assert config.experiment_name is None
        assert config.downgrade_unity_catalog is True
        assert config.model_dir == "models"
        assert config.config_path == "config.yaml"

    def test_mlflow_tracker_instance_config_init_with_fields(self) -> None:
        """Test MlflowTrackerInstanceConfig instantiation with fields."""
        config = MlflowTrackerInstanceConfig(
            experiment_name="my_experiment",
            run_name="my_run",
            tracking_uri="databricks",
        )
        assert config.experiment_name == "my_experiment"
        assert config.run_name == "my_run"
        assert config.tracking_uri == "databricks"
        assert config._target_ == "simplexity.tracking.mlflow_tracker.MlflowTracker"

    def test_validate_tracking_config_with_mlflow_tracker(self) -> None:
        """Test validate_tracking_config with MlflowTracker instance."""
        # Valid mlflow tracker config
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                    }
                )
            }
        )
        validate_tracking_config(cfg)


class TestS3TrackerConfig:
    """Test S3Tracker configuration functions."""

    def test_is_s3_tracker_target_valid(self) -> None:
        """Test is_s3_tracker_target with valid target."""
        assert is_s3_tracker_target("simplexity.tracking.s3_tracker.S3Tracker.from_config")

    def test_is_s3_tracker_target_invalid(self) -> None:
        """Test is_s3_tracker_target with invalid targets."""
        assert not is_s3_tracker_target("simplexity.tracking.s3_tracker.S3Tracker")
        assert not is_s3_tracker_target("simplexity.tracking.file_tracker.FileTracker")
        assert not is_s3_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker")
        assert not is_s3_tracker_target("")
        assert not is_s3_tracker_target("some.other.target")

    def test_is_s3_tracker_config_valid(self) -> None:
        """Test is_s3_tracker_config with valid configs."""
        cfg = DictConfig({"_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config"})
        assert is_s3_tracker_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                "prefix": "s3://bucket/prefix",
            }
        )
        assert is_s3_tracker_config(cfg)

    def test_is_s3_tracker_config_invalid(self) -> None:
        """Test is_s3_tracker_config with invalid configs."""
        # Non-s3 tracker target
        cfg = DictConfig({"_target_": "simplexity.tracking.file_tracker.FileTracker"})
        assert not is_s3_tracker_config(cfg)

        # Missing _target_
        cfg = DictConfig({"prefix": "s3://bucket/prefix"})
        assert not is_s3_tracker_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_s3_tracker_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_s3_tracker_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_s3_tracker_config(cfg)

    def test_validate_s3_tracker_instance_config_valid(self) -> None:
        """Test validate_s3_tracker_instance_config with valid configs."""
        # Valid config with required fields and config_filename
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                "prefix": "s3://bucket/prefix",
                "config_filename": "config.ini",
            }
        )
        validate_s3_tracker_instance_config(cfg)  # Should not raise

        # Valid config with optional config_filename
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                "prefix": "s3://bucket/prefix",
                "config_filename": "custom.ini",
            }
        )
        validate_s3_tracker_instance_config(cfg)  # Should not raise

    def test_validate_s3_tracker_instance_config_invalid_target(self) -> None:
        """Test validate_s3_tracker_instance_config raises with invalid target."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "prefix": "s3://bucket/prefix",
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be"):
            validate_s3_tracker_instance_config(cfg)

    def test_validate_s3_tracker_instance_config_missing_prefix(self) -> None:
        """Test validate_s3_tracker_instance_config raises when prefix is missing."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
            }
        )
        with pytest.raises(ConfigValidationError, match="S3TrackerInstanceConfig.prefix must be a string"):
            validate_s3_tracker_instance_config(cfg)

    def test_validate_s3_tracker_instance_config_empty_prefix(self) -> None:
        """Test validate_s3_tracker_instance_config raises when prefix is empty."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                "prefix": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="S3TrackerInstanceConfig.prefix must be a non-empty string"):
            validate_s3_tracker_instance_config(cfg)

    def test_validate_s3_tracker_instance_config_empty_config_filename(self) -> None:
        """Test validate_s3_tracker_instance_config raises when config_filename is empty."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                "prefix": "s3://bucket/prefix",
                "config_filename": "",
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="S3TrackerInstanceConfig.config_filename must be a non-empty string",
        ):
            validate_s3_tracker_instance_config(cfg)

    def test_s3_tracker_instance_config_init(self) -> None:
        """Test S3TrackerInstanceConfig instantiation."""
        config = S3TrackerInstanceConfig(prefix="s3://bucket/prefix")
        assert config.prefix == "s3://bucket/prefix"
        assert config._target_ == "simplexity.tracking.s3_tracker.S3Tracker.from_config"
        assert config.config_filename == "config.ini"  # Default value

    def test_s3_tracker_instance_config_init_with_custom_config_filename(self) -> None:
        """Test S3TrackerInstanceConfig instantiation with custom config_filename."""
        config = S3TrackerInstanceConfig(prefix="s3://bucket/prefix", config_filename="custom.ini")
        assert config.prefix == "s3://bucket/prefix"
        assert config.config_filename == "custom.ini"
        assert config._target_ == "simplexity.tracking.s3_tracker.S3Tracker.from_config"

    def test_validate_tracking_config_with_s3_tracker(self) -> None:
        """Test validate_tracking_config with S3Tracker instance."""
        # Valid s3 tracker config
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                        "prefix": "s3://bucket/prefix",
                        "config_filename": "config.ini",
                    }
                )
            }
        )
        validate_tracking_config(cfg)

        # Missing prefix
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="S3TrackerInstanceConfig.prefix must be a string"):
            validate_tracking_config(cfg)

        # Empty prefix
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config",
                        "prefix": "",
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="S3TrackerInstanceConfig.prefix must be a non-empty string"):
            validate_tracking_config(cfg)


class TestRunTrackerConfig:
    """Test run tracker configuration functions."""

    def test_is_run_tracker_target_valid(self) -> None:
        """Test is_run_tracker_target with valid targets."""
        assert is_run_tracker_target("simplexity.tracking.file_tracker.FileTracker")
        assert is_run_tracker_target("simplexity.tracking.mlflow_tracker.MlflowTracker")
        assert is_run_tracker_target("simplexity.tracking.s3_tracker.S3Tracker.from_config")
        assert is_run_tracker_target("simplexity.tracking.any_tracker.AnyTracker")

    def test_is_run_tracker_target_invalid(self) -> None:
        """Test is_run_tracker_target with invalid targets."""
        assert not is_run_tracker_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert not is_run_tracker_target("torch.optim.Adam")
        assert not is_run_tracker_target("")
        assert not is_run_tracker_target("some.other.target")
        assert not is_run_tracker_target("simplexity.logging.mlflow_logger.MLFlowLogger")

    def test_is_run_tracker_config_valid(self) -> None:
        """Test is_run_tracker_config with valid configs."""
        cfg = DictConfig({"_target_": "simplexity.tracking.file_tracker.FileTracker"})
        assert is_run_tracker_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker"})
        assert is_run_tracker_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.tracking.s3_tracker.S3Tracker.from_config"})
        assert is_run_tracker_config(cfg)

    def test_is_run_tracker_config_invalid(self) -> None:
        """Test is_run_tracker_config with invalid configs."""
        # Non-tracker target
        cfg = DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})
        assert not is_run_tracker_config(cfg)

        # Missing _target_
        cfg = DictConfig({"experiment_name": "my_experiment"})
        assert not is_run_tracker_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_run_tracker_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_run_tracker_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_run_tracker_config(cfg)


class TestUpdateTrackingInstanceConfig:
    """Test update_tracking_instance_config function."""

    def test_update_tracking_instance_config(self) -> None:
        """Test update_tracking_instance_config function."""
        # Initial config
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_name": "exp1",
                "run_name": "run1",
            }
        )

        # Update config
        updated_cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.mlflow_tracker.MlflowTracker",
                "experiment_name": "exp2",
                "tracking_uri": "file:///tmp/mlruns",
            }
        )

        update_tracking_instance_config(cfg, updated_cfg)

        assert cfg.experiment_name == "exp2"
        assert cfg.run_name == "run1"  # Should remain unchanged
        assert cfg.tracking_uri == "file:///tmp/mlruns"

    def test_update_tracking_instance_config_overwrites(self) -> None:
        """Test update_tracking_instance_config overwrites existing values."""
        cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "/tmp/old.log",
                "model_dir_name": "old_models",
            }
        )

        updated_cfg = DictConfig(
            {
                "_target_": "simplexity.tracking.file_tracker.FileTracker",
                "file_path": "/tmp/new.log",
            }
        )

        update_tracking_instance_config(cfg, updated_cfg)

        assert cfg.file_path == "/tmp/new.log"
        assert cfg.model_dir_name == "old_models"  # Should remain unchanged
