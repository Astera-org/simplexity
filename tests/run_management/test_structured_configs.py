import pytest
from omegaconf import MISSING, DictConfig

from simplexity.exceptions import ConfigValidationError
from simplexity.run_management.structured_configs import (
    is_generative_process_config,
    is_generative_process_target,
    is_logger_config,
    is_logger_target,
    validate_generative_process_config,
    validate_logging_config,
    validate_mlflow_config,
)


def test_validate_mlflow_config_valid():
    """Test validate_mlflow_config with valid configs."""
    cfg = DictConfig(
        {
            "experiment_name": "my_experiment",
            "run_name": "my_run",
        }
    )
    validate_mlflow_config(cfg)

    cfg = DictConfig(
        {
            "experiment_name": "my_experiment",
            "run_name": "my_run",
            "tracking_uri": "databricks",
            "registry_uri": "databricks",
            "downgrade_unity_catalog": True,
        }
    )
    validate_mlflow_config(cfg)


def test_validate_mlflow_config_invalid():
    """Test validate_mlflow_config with invalid configs."""
    cfg = DictConfig(
        {
            "run_name": "my_run",
            "tracking_uri": "databricks",
            "registry_uri": "databricks",
            "downgrade_unity_catalog": True,
        }
    )
    with pytest.raises(ConfigValidationError, match="MLFlowConfig.experiment_name must be a non-empty string"):
        validate_mlflow_config(cfg)

    cfg = DictConfig(
        {
            "experiment_name": "my_experiment",
            "tracking_uri": "databricks",
            "registry_uri": "databricks",
            "downgrade_unity_catalog": True,
        }
    )
    with pytest.raises(ConfigValidationError, match="MLFlowConfig.run_name must be a non-empty string"):
        validate_mlflow_config(cfg)


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


def test_is_generative_process_target_valid():
    """Test is_generative_process_target with valid generative process targets."""
    assert is_generative_process_target("simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel")
    assert is_generative_process_target("simplexity.generative_processes.builder.build_hidden_markov_model")


def test_is_generative_process_target_invalid():
    """Test is_generative_process_target with invalid targets."""
    assert not is_generative_process_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
    assert not is_generative_process_target("torch.optim.Adam")
    assert not is_generative_process_target("")


def test_is_generative_process_config_valid():
    """Test is_generative_process_config with valid generative process configs."""
    cfg = DictConfig({"_target_": "simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel"})
    assert is_generative_process_config(cfg)

    cfg = DictConfig(
        {
            "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
            "process_name": "mess3",
            "x": 0.15,
            "a": 0.6,
        }
    )
    assert is_generative_process_config(cfg)


def test_is_generative_process_config_invalid():
    """Test is_generative_process_config with invalid configs."""
    # Non-generative process target
    cfg = DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})
    assert not is_generative_process_config(cfg)

    # Missing _target_
    cfg = DictConfig({"process_name": "mess3", "x": 0.15, "a": 0.6})
    assert not is_generative_process_config(cfg)

    # _target_ is not a omegaconf target
    cfg = DictConfig({"target": "simplexity.generative_processes.builder.build_hidden_markov_model"})
    assert not is_generative_process_config(cfg)

    # _target_ is None
    cfg = DictConfig({"_target_": None})
    assert not is_generative_process_config(cfg)

    # _target_ is not a string
    cfg = DictConfig({"_target_": 123})
    assert not is_generative_process_config(cfg)

    # Empty config
    cfg = DictConfig({})
    assert not is_generative_process_config(cfg)


def test_validate_generative_process_config_valid():
    """Test validate_generative_process_config with valid configs."""
    cfg = DictConfig(
        {
            "instance": DictConfig(
                {
                    "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                    "process_name": "mess3",
                    "x": 0.15,
                    "a": 0.6,
                }
            ),
            "base_vocab_size": MISSING,
            "vocab_size": MISSING,
        }
    )
    validate_generative_process_config(cfg)

    cfg = DictConfig(
        {
            "instance": DictConfig(
                {
                    "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
                    "process_name": "mess3",
                    "x": 0.15,
                    "a": 0.6,
                }
            ),
            "name": "mess3",
            "base_vocab_size": 3,
            "bos_token": 3,
            "eos_token": None,
            "vocab_size": 4,
        }
    )
    validate_generative_process_config(cfg)


def test_validate_generative_process_config_missing_instance():
    """Test validate_generative_process_config raises when instance is missing."""
    cfg = DictConfig({})
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.instance is required"):
        validate_generative_process_config(cfg)

    cfg = DictConfig(
        {
            "_target_": "simplexity.generative_processes.builder.build_hidden_markov_model",
            "process_name": "mess3",
            "x": 0.15,
            "a": 0.6,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.instance is required"):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_invalid_instance():
    """Test validate_generative_process_config raises when instance is invalid."""
    # Instance without _target_
    cfg = DictConfig({"instance": DictConfig({"process_name": "mess3", "x": 0.15, "a": 0.6})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
        validate_generative_process_config(cfg)

    # Instance with empty _target_
    cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
        validate_generative_process_config(cfg)

    # Instance with non-string _target_
    cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
    with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_non_generative_process_target():
    """Test validate_generative_process_config raises when instance target is not a generative process target."""
    cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})})
    with pytest.raises(
        ConfigValidationError, match="GenerativeProcessConfig.instance._target_ must be a generative process target"
    ):
        validate_generative_process_config(cfg)

    cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
    with pytest.raises(
        ConfigValidationError, match="GenerativeProcessConfig.instance._target_ must be a generative process target"
    ):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_invalid_name():
    """Test validate_generative_process_config raises when name is invalid."""
    # Empty string name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "",
            "base_vocab_size": MISSING,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be None or a non-empty string"):
        validate_generative_process_config(cfg)

    # Whitespace-only name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "   ",
            "base_vocab_size": MISSING,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be None or a non-empty string"):
        validate_generative_process_config(cfg)

    # Non-string name
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": 123,
            "base_vocab_size": MISSING,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be None or a non-empty string"):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_invalid_base_vocab_size():
    """Test validate_generative_process_config raises when base_vocab_size is invalid."""
    # Non-integer base_vocab_size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": "3",
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be positive"):
        validate_generative_process_config(cfg)

    # Negative base_vocab_size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": -1,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be positive"):
        validate_generative_process_config(cfg)


@pytest.mark.parametrize("token_type", ["bos_token", "eos_token"])
def test_validate_generative_process_config_invalid_special_tokens(token_type: str):
    """Test validate_generative_process_config raises when special tokens are invalid."""
    # Non-integer token value
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            token_type: "3",
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.bos_token must be an int or None"):
        validate_generative_process_config(cfg)

    # Negative token value
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            token_type: -1,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.bos_token must be non-negative"):
        validate_generative_process_config(cfg)

    # Token value greater than vocab size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            token_type: 4,
            "vocab_size": 4,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.eos_token must be non-negative"):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_invalid_bos_eos_token_same_value():
    """Test validate_generative_process_config raises when bos_token and eos_token are the same."""
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            "bos_token": 3,
            "eos_token": 3,
            "vocab_size": MISSING,
        }
    )
    with pytest.raises(
        ConfigValidationError, match="GenerativeProcessConfig.bos_token and eos_token cannot be the same"
    ):
        validate_generative_process_config(cfg)


def test_validate_generative_process_config_invalid_vocab_size():
    """Test validate_generative_process_config raises when vocab_size is invalid."""
    # Non-integer vocab size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            "bos_token": 3,
            "eos_token": None,
            "vocab_size": "4",
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.vocab_size must be an int"):
        validate_generative_process_config(cfg)

    # Negative vocab size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            "bos_token": 3,
            "eos_token": None,
            "vocab_size": -1,
        }
    )
    with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.vocab_size must be positive"):
        validate_generative_process_config(cfg)

    # Incorrect vocab size
    cfg = DictConfig(
        {
            "instance": DictConfig({"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}),
            "name": "mess3",
            "base_vocab_size": 3,
            "bos_token": 3,
            "eos_token": None,
            "vocab_size": 3,
        }
    )
    with pytest.raises(ConfigValidationError):
        validate_generative_process_config(cfg)
