import pytest
from omegaconf import DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.run_management.structured_configs import is_logger_config, is_logger_target, validate_logging_config


def test_is_logger_target_valid():
    """Test is_logger_target with valid logger targets."""
    assert is_logger_target("simplexity.logging.file_logger.FileLogger")
    assert is_logger_target("simplexity.logging.mlflow_logger.MLFlowLogger")
    assert is_logger_target("simplexity.logging.print_logger.PrintLogger")


def test_is_logger_target_invalid():
    """Test is_logger_target with invalid targets."""
    assert not is_logger_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
    assert not is_logger_target("logging.Logger")
    assert not is_logger_target("")


def test_is_logger_config_valid():
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


def test_is_logger_config_invalid():
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


def test_validate_logging_config_valid():
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


def test_validate_logging_config_missing_instance():
    """Test validate_logging_config raises when instance is missing."""
    cfg = DictConfig({})
    with pytest.raises(ConfigValidationError, match="LoggingConfig.instance is required"):
        validate_logging_config(cfg)

    cfg = DictConfig({"name": "my_logger"})
    with pytest.raises(ConfigValidationError, match="LoggingConfig.instance is required"):
        validate_logging_config(cfg)


def test_validate_logging_config_invalid_instance():
    """Test validate_logging_config raises when instance is invalid."""
    # Instance without _target_
    cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
        validate_logging_config(cfg)

    # Instance with empty _target_
    cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty"):
        validate_logging_config(cfg)

    # Instance with non-string _target_
    cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
        validate_logging_config(cfg)


def test_validate_logging_config_non_logger_target():
    """Test validate_logging_config raises when instance target is not a logger target."""
    cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})})
    with pytest.raises(ConfigValidationError, match="LoggingConfig.instance._target_ must be a logger target"):
        validate_logging_config(cfg)

    cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
    with pytest.raises(ConfigValidationError, match="LoggingConfig.instance._target_ must be a logger target"):
        validate_logging_config(cfg)


def test_validate_logging_config_invalid_name():
    """Test validate_logging_config raises when name is invalid."""
    # Empty string name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
            "name": "",
        }
    )
    with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be None or a non-empty string"):
        validate_logging_config(cfg)

    # Whitespace-only name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
            "name": "   ",
        }
    )
    with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be None or a non-empty string"):
        validate_logging_config(cfg)

    # Non-string name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"}),
            "name": 123,
        }
    )
    with pytest.raises(ConfigValidationError, match="LoggingConfig.name must be None or a non-empty string"):
        validate_logging_config(cfg)
