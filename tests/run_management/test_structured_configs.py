import re
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from omegaconf import MISSING, DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.run_management.structured_configs import (
    GenerativeProcessConfig,
    HookedTransformerConfig,
    HookedTransformerConfigConfig,
    InstanceConfig,
    LoggingConfig,
    MLFlowConfig,
    OptimizerConfig,
    PersistenceConfig,
    PredictiveModelConfig,
    is_generative_process_config,
    is_generative_process_target,
    is_hooked_transformer_config,
    is_logger_config,
    is_logger_target,
    is_model_persister_target,
    is_optimizer_config,
    is_optimizer_target,
    is_persister_config,
    is_predictive_model_config,
    is_predictive_model_target,
    is_pytorch_optimizer_config,
    validate_base_config,
    validate_generative_process_config,
    validate_hooked_transformer_config,
    validate_hooked_transformer_config_config,
    validate_logging_config,
    validate_mlflow_config,
    validate_optimizer_config,
    validate_persistence_config,
    validate_predictive_model_config,
)


class TestInstanceConfig:
    """Test InstanceConfig."""

    def test_instance_config(self):
        """Test creating instance config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(InstanceConfig(_target_="some_target"))
        assert cfg.get("_target_") == "some_target"

    def test_instance_derived_config(self):
        """Test creating a child instance config from dataclass."""

        @dataclass
        class SomeInstance(InstanceConfig):
            """Some instance config."""

            other_key: str
            default_attribute: int = 42

        cfg: DictConfig = OmegaConf.structured(
            SomeInstance(
                _target_="some_target",
                other_key="other_value",
            )
        )
        assert cfg.get("_target_") == "some_target"
        assert cfg.get("other_key") == "other_value"
        assert cfg.get("default_attribute") == 42


# ============================================================================
# MLFlow Config Tests
# ============================================================================


class TestMLFlowConfig:
    def test_mlflow_config(self):
        """Test creating mlflow config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(MLFlowConfig(experiment_name="some_experiment", run_name="some_run"))
        assert cfg.get("experiment_name") == "some_experiment"
        assert cfg.get("run_name") == "some_run"
        assert cfg.get("tracking_uri") is None
        assert cfg.get("registry_uri") is None
        assert cfg.get("downgrade_unity_catalog") is None

    def test_validate_mlflow_config_valid(self):
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

    def test_validate_mlflow_config_missing_required_fields(self):
        """Test validate_mlflow_config with missing required fields."""
        # Missing experiment_name
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

        # Missing run_name
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

    def test_validate_mlflow_config_invalid_downgrade_unity_catalog(self):
        """Test validate_mlflow_config with invalid downgrade_unity_catalog."""
        cfg = DictConfig(
            {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                "downgrade_unity_catalog": "not_a_bool",
            }
        )
        with pytest.raises(ConfigValidationError, match="MLFlowConfig.downgrade_unity_catalog must be a bool"):
            validate_mlflow_config(cfg)

    @pytest.mark.parametrize("uri_type", ["tracking_uri", "registry_uri"])
    def test_validate_mlflow_config_invalid_uri(self, uri_type: str):
        """Test validate_mlflow_config with invalid tracking_uri and registry_uri."""
        # Empty URI
        cfg = DictConfig(
            {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                uri_type: "  ",
            }
        )
        with pytest.raises(ConfigValidationError, match=f"MLFlowConfig.{uri_type} cannot be empty"):
            validate_mlflow_config(cfg)

        # parse error (urlparse raises an exception)
        cfg = DictConfig(
            {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                uri_type: "%parse_error%",
            }
        )
        with pytest.raises(ConfigValidationError, match=f"MLFlowConfig.{uri_type} is not a valid URI"):
            validate_mlflow_config(cfg)

        # missing scheme
        cfg = DictConfig(
            {
                "experiment_name": "my_experiment",
                "run_name": "my_run",
                uri_type: "no_scheme",
            }
        )
        with pytest.raises(ConfigValidationError, match=f"MLFlowConfig.{uri_type} must have a valid URI scheme"):
            validate_mlflow_config(cfg)


# ============================================================================
# Logger Config Tests
# ============================================================================


class TestLoggerConfig:
    def test_logging_config(self):
        """Test creating logger config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(LoggingConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_logger_target_valid(self):
        """Test is_logger_target with valid logger targets."""
        assert is_logger_target("simplexity.logging.file_logger.FileLogger")
        assert is_logger_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert is_logger_target("simplexity.logging.print_logger.PrintLogger")

    def test_is_logger_target_invalid(self):
        """Test is_logger_target with invalid targets."""
        assert not is_logger_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert not is_logger_target("logging.Logger")
        assert not is_logger_target("")

    def test_is_logger_config_valid(self):
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

    def test_is_logger_config_invalid(self):
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

    def test_validate_logging_config_valid(self):
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

    def test_validate_logging_config_missing_instance(self):
        """Test validate_logging_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance is required"):
            validate_logging_config(cfg)

        cfg = DictConfig({"name": "my_logger"})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance is required"):
            validate_logging_config(cfg)

    def test_validate_logging_config_invalid_instance(self):
        """Test validate_logging_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_logging_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
            validate_logging_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_logging_config(cfg)

    def test_validate_logging_config_non_logger_target(self):
        """Test validate_logging_config raises when instance target is not a logger target."""
        cfg = DictConfig(
            {"instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})}
        )
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance._target_ must be a logger target"):
            validate_logging_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(ConfigValidationError, match="LoggingConfig.instance._target_ must be a logger target"):
            validate_logging_config(cfg)

    def test_validate_logging_config_invalid_name(self):
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


# ============================================================================
# Generative Process Config Tests
# ============================================================================


class TestGenerativeProcessConfig:
    """Test GenerativeProcessConfig."""

    def test_generative_process_config(self):
        """Test creating generative process config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(
            GenerativeProcessConfig(
                instance=InstanceConfig(_target_="some_target"),
                base_vocab_size=3,
                bos_token=None,
                eos_token=None,
                vocab_size=3,
            )
        )
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None
        assert cfg.get("base_vocab_size") == 3
        assert cfg.get("bos_token") is None
        assert cfg.get("eos_token") is None
        assert cfg.get("vocab_size") == 3
        assert cfg.get("sequence_len") is None
        assert cfg.get("batch_size") is None

    def test_is_generative_process_target_valid(self):
        """Test is_generative_process_target with valid generative process targets."""
        assert is_generative_process_target("simplexity.generative_processes.hidden_markov_model.HiddenMarkovModel")
        assert is_generative_process_target("simplexity.generative_processes.builder.build_hidden_markov_model")

    def test_is_generative_process_target_invalid(self):
        """Test is_generative_process_target with invalid targets."""
        assert not is_generative_process_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert not is_generative_process_target("torch.optim.Adam")
        assert not is_generative_process_target("")

    def test_is_generative_process_config_valid(self):
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

    def test_is_generative_process_config_invalid(self):
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

    def test_validate_generative_process_config_valid(self):
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
                "sequence_len": 256,
                "batch_size": 64,
            }
        )
        validate_generative_process_config(cfg)

    def test_validate_generative_process_config_missing_instance(self):
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
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.instance is required"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_instance(self):
        """Test validate_generative_process_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig(
            {
                "instance": DictConfig({"process_name": "mess3", "x": 0.15, "a": 0.6}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_generative_process_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""}), "base_vocab_size": MISSING, "vocab_size": MISSING})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
            validate_generative_process_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123}), "base_vocab_size": MISSING, "vocab_size": MISSING})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_non_generative_process_target(self):
        """Test validate_generative_process_config raises when instance target is not a generative process target."""
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="GenerativeProcessConfig.instance._target_ must be a generative process target"
        ):
            validate_generative_process_config(cfg)

        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.optim.Adam"}),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match="GenerativeProcessConfig.instance._target_ must be a generative process target",
        ):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_name(self):
        """Test validate_generative_process_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "",
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a non-empty string"):
            validate_generative_process_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "   ",
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a non-empty string"):
            validate_generative_process_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": 123,
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.name must be a string or None"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_base_vocab_size(self):
        """Test validate_generative_process_config raises when base_vocab_size is invalid."""
        # Non-integer base_vocab_size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": "3",
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be an int"):
            validate_generative_process_config(cfg)

        # Negative base_vocab_size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": -1,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.base_vocab_size must be positive"):
            validate_generative_process_config(cfg)

    @pytest.mark.parametrize("token_type", ["bos_token", "eos_token"])
    def test_validate_generative_process_config_invalid_special_tokens(self, token_type: str):
        """Test validate_generative_process_config raises when special tokens are invalid."""
        # Non-integer token value
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: "3",
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match=re.escape(f"GenerativeProcessConfig.{token_type} must be an int or None, got <class 'str'>"),
        ):
            validate_generative_process_config(cfg)

        # Negative token value
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: -1,
                "vocab_size": MISSING,
            }
        )
        with pytest.raises(ConfigValidationError, match=f"GenerativeProcessConfig.{token_type} must be non-negative"):
            validate_generative_process_config(cfg)

        # Token value greater than vocab size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": 3,
                token_type: 4,
                "vocab_size": 4,
            }
        )
        with pytest.raises(
            ConfigValidationError, match=re.escape(f"GenerativeProcessConfig.{token_type} (4) must be < vocab_size (4)")
        ):
            validate_generative_process_config(cfg)

    @pytest.mark.parametrize("attribute", ["base_vocab_size", "bos_token", "eos_token", "vocab_size"])
    def test_validate_generative_process_config_missing_special_tokens(self, attribute: str):
        """Test validate_generative_process_config raises when special tokens are missing."""
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "base_vocab_size": 3,
                "vocab_size": 4,
            }
        )
        cfg[attribute] = MISSING
        # assert that there is a SIMPLEXITY_LOGGER debug log
        with patch("simplexity.run_management.structured_configs.SIMPLEXITY_LOGGER.debug") as mock_debug:
            validate_generative_process_config(cfg)
            mock_debug.assert_called_once_with(
                f"[generative process] {attribute} is missing, will be resolved dynamically"
            )

    def test_validate_generative_process_config_invalid_bos_eos_token_same_value(self):
        """Test validate_generative_process_config raises when bos_token and eos_token are the same."""
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
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

    def test_validate_generative_process_config_invalid_vocab_size(self):
        """Test validate_generative_process_config raises when vocab_size is invalid."""
        # Non-integer vocab size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
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
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
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
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "name": "mess3",
                "base_vocab_size": 3,
                "bos_token": 3,
                "eos_token": None,
                "vocab_size": 5,
            }
        )
        with pytest.raises(
            ConfigValidationError,
            match=re.escape(
                "GenerativeProcessConfig.vocab_size (5) must be equal to "
                "base_vocab_size (3) + use_bos_token (True) + use_eos_token (False) = 4"
            ),
        ):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_sequence_len(self):
        """Test validate_generative_process_config raises when sequence_len is invalid."""
        # Non-integer sequence_len
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
                "sequence_len": "16",
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.sequence_len must be an int"):
            validate_generative_process_config(cfg)

        # Negative sequence_len
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
                "sequence_len": -1,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.sequence_len must be positive"):
            validate_generative_process_config(cfg)

    def test_validate_generative_process_config_invalid_batch_size(self):
        """Test validate_generative_process_config raises when batch_size is invalid."""
        # Non-integer batch_size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
                "batch_size": "64",
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.batch_size must be an int"):
            validate_generative_process_config(cfg)

        # Negative batch_size
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {"_target_": "simplexity.generative_processes.builder.build_hidden_markov_model"}
                ),
                "base_vocab_size": MISSING,
                "vocab_size": MISSING,
                "batch_size": -1,
            }
        )
        with pytest.raises(ConfigValidationError, match="GenerativeProcessConfig.batch_size must be positive"):
            validate_generative_process_config(cfg)


# ============================================================================
# Persistence Config Tests
# ============================================================================


class TestPersistenceConfig:
    """Test PersistenceConfig."""

    def test_persistence_config(self):
        """Test creating persistence config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(PersistenceConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_model_persister_target_valid(self):
        """Test is_model_persister_target with valid persister targets."""
        assert is_model_persister_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert is_model_persister_target("simplexity.persistence.local_pytorch_persister.LocalPytorchPersister")

    def test_is_model_persister_target_invalid(self):
        """Test is_model_persister_target with invalid targets."""
        assert not is_model_persister_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_model_persister_target("torch.optim.Adam")
        assert not is_model_persister_target("")

    def test_is_persister_config_valid(self):
        """Test is_persister_config with valid persister configs."""
        cfg = DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"})
        assert is_persister_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister",
                "path": "/tmp/model",
            }
        )
        assert is_persister_config(cfg)

    def test_is_persister_config_invalid(self):
        """Test is_persister_config with invalid configs."""
        # Non-persister target
        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_persister_config(cfg)

        # Missing _target_
        cfg = DictConfig({"path": "/tmp/model"})
        assert not is_persister_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_persister_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_persister_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_persister_config(cfg)

    def test_validate_persistence_config_valid(self):
        """Test validate_persistence_config with valid configs."""
        # Valid config without name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                    }
                ),
            }
        )
        validate_persistence_config(cfg)  # Should not raise

        # Valid config with name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                    }
                ),
                "name": "my_persister",
            }
        )
        validate_persistence_config(cfg)  # Should not raise

        # Valid config with None name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister",
                        "experiment_name": "my_experiment",
                        "run_name": "my_run",
                    }
                ),
                "name": None,
            }
        )
        validate_persistence_config(cfg)  # Should not raise

    def test_validate_persistence_config_missing_instance(self):
        """Test validate_persistence_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance is required"):
            validate_persistence_config(cfg)

        cfg = DictConfig({"name": "my_persister"})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance is required"):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_invalid_instance(self):
        """Test validate_persistence_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_persistence_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
            validate_persistence_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_non_persister_target(self):
        """Test validate_persistence_config raises when instance target is not a persister target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(
            ConfigValidationError, match="PersistenceConfig.instance._target_ must be a persister target"
        ):
            validate_persistence_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(
            ConfigValidationError, match="PersistenceConfig.instance._target_ must be a persister target"
        ):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_invalid_name(self):
        """Test validate_persistence_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.name must be a non-empty string"):
            validate_persistence_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.name must be a non-empty string"):
            validate_persistence_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "simplexity.persistence.mlflow_persister.MLFlowPersister"}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.name must be a string or None"):
            validate_persistence_config(cfg)


# ============================================================================
# Predictive Model Config Tests
# ============================================================================


class TestHookedTransformerConfig:
    """Test PredictiveModelConfig."""

    def test_hooked_transformer_config(self):
        """Test creating hooked transformer config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(
            PredictiveModelConfig(
                instance=HookedTransformerConfig(
                    cfg=HookedTransformerConfigConfig(
                        n_layers=2,
                        d_model=128,
                        d_head=32,
                        n_ctx=16,
                        d_vocab=3,
                    ),
                )
            )
        )
        assert OmegaConf.select(cfg, "instance._target_") == "transformer_lens.HookedTransformer"
        assert OmegaConf.select(cfg, "instance.cfg._target_") == "transformer_lens.HookedTransformerConfig"
        assert OmegaConf.select(cfg, "instance.cfg.n_layers") == 2
        assert OmegaConf.select(cfg, "instance.cfg.d_model") == 128
        assert OmegaConf.select(cfg, "instance.cfg.d_head") == 32
        assert OmegaConf.select(cfg, "instance.cfg.n_ctx") == 16
        assert OmegaConf.select(cfg, "instance.cfg.n_heads") == -1
        assert OmegaConf.select(cfg, "instance.cfg.d_mlp") is None
        assert OmegaConf.select(cfg, "instance.cfg.d_vocab") == 3
        assert OmegaConf.select(cfg, "instance.cfg.act_fn") is None
        assert OmegaConf.select(cfg, "instance.cfg.normalization_type") == "LN"
        assert OmegaConf.select(cfg, "instance.cfg.device") is None
        assert OmegaConf.select(cfg, "instance.cfg.seed") is None
        assert cfg.get("name") is None
        assert cfg.get("load_checkpoint_step") is None

    def test_validate_hooked_transformer_config_config_valid(self):
        """Test validate_hooked_transformer_config_config with valid configs."""
        # Test with minimal config (n_heads defaults to -1)
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": MISSING,
                "n_heads": -1,
                "d_vocab": MISSING,
            }
        )
        validate_hooked_transformer_config_config(cfg)

        # Test with full config including n_heads
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": 256,
                "n_heads": -1,
                "d_mlp": 512,
                "act_fn": None,
                "d_vocab": 1000,
                "normalization_type": None,
                "device": None,
                "seed": 42,
            }
        )
        validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_incompatible_dimensions(self):
        """Test validate_hooked_transformer_config_config raises when n_heads=-1.

        Specifically tests that d_model must be divisible by d_head when n_heads=-1.
        """
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 130,  # Not divisible by d_head (130 % 32 == 2)
                "d_head": 32,
                "n_ctx": MISSING,
                "n_heads": -1,
                "d_mlp": 512,
                "act_fn": None,
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": None,
                "seed": None,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_model.*must be divisible by d_head"):
            validate_hooked_transformer_config_config(cfg)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("d_model", 130),
            ("d_head", 32),
            ("n_heads", -1),
            ("n_layers", 2),
            ("d_mlp", 512),
            ("d_vocab", 1000),
            ("n_ctx", 256),
        ],
    )
    def test_validate_hooked_transformer_config_config_invalid_fields(self, field, value):
        """Test validate_hooked_transformer_config_config raises for invalid field values."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 32,
                "n_ctx": MISSING,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )

        # Non-integer value
        cfg[field] = str(value)
        with pytest.raises(ConfigValidationError, match=f"HookedTransformerConfigConfig.{field} must be an int"):
            validate_hooked_transformer_config_config(cfg)

        # Non-positive value (0 should fail for all fields, including n_heads)
        cfg[field] = 0
        with pytest.raises(ConfigValidationError, match=f"HookedTransformerConfigConfig.{field} must be positive"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_d_model_not_divisible_by_n_heads(self):
        """Test validate_hooked_transformer_config_config raises when d_model is not divisible by n_heads."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 130,
                "d_head": 32,
                "n_ctx": MISSING,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_model.*must be divisible by n_heads"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_config_d_head_times_n_heads_not_equal_d_model(self):
        """Test validate_hooked_transformer_config_config raises when d_head * n_heads != d_model."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformerConfig",
                "n_layers": 2,
                "d_model": 128,
                "d_head": 30,
                "n_ctx": MISSING,
                "n_heads": 4,
                "d_mlp": 512,
                "act_fn": "relu",
                "d_vocab": MISSING,
                "normalization_type": "LN",
                "device": "cpu",
                "seed": 42,
            }
        )
        with pytest.raises(ConfigValidationError, match="d_head.*n_heads.*must equal d_model"):
            validate_hooked_transformer_config_config(cfg)

    def test_validate_hooked_transformer_config_valid(self):
        """Test validate_hooked_transformer_config with valid configs."""
        cfg = DictConfig(
            {
                "_target_": "transformer_lens.HookedTransformer",
                "cfg": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformerConfig",
                        "n_layers": 2,
                        "d_model": 128,
                        "d_head": 32,
                        "n_ctx": MISSING,
                        "n_heads": 4,
                        "d_mlp": 512,
                        "act_fn": "relu",
                        "d_vocab": MISSING,
                        "normalization_type": "LN",
                        "device": "cpu",
                        "seed": 42,
                    }
                ),
            }
        )
        validate_hooked_transformer_config(cfg)

    def test_validate_hooked_transformer_config_missing_target(self):
        """Test validate_hooked_transformer_config raises when _target_ is missing."""
        cfg = DictConfig(
            {
                "cfg": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformerConfig",
                        "n_layers": 2,
                        "d_model": 128,
                        "d_head": 32,
                        "n_ctx": MISSING,
                        "n_heads": 4,
                        "d_mlp": 512,
                        "act_fn": "relu",
                        "d_vocab": MISSING,
                        "normalization_type": "LN",
                        "device": "cpu",
                        "seed": 42,
                    }
                ),
            }
        )
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_hooked_transformer_config(cfg)

    def test_validate_hooked_transformer_config_missing_cfg(self):
        """Test validate_hooked_transformer_config raises when cfg is missing."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        with pytest.raises(ConfigValidationError, match="HookedTransformerConfig.cfg is required"):
            validate_hooked_transformer_config(cfg)

    def test_is_predictive_model_target_valid(self):
        """Test is_predictive_model_target with valid model targets."""
        assert is_predictive_model_target("transformer_lens.HookedTransformer")
        assert is_predictive_model_target("torch.nn.Linear")
        assert is_predictive_model_target("equinox.nn.Linear")
        assert is_predictive_model_target("penzai.nn.Linear")
        assert is_predictive_model_target("penzai.models.Transformer")
        assert is_predictive_model_target("simplexity.predictive_models.MyModel")

    def test_is_predictive_model_target_invalid(self):
        """Test is_predictive_model_target with invalid targets."""
        assert not is_predictive_model_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_predictive_model_target("torch.optim.Adam")
        assert not is_predictive_model_target("")

    def test_is_predictive_model_config_valid(self):
        """Test is_predictive_model_config with valid predictive model configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        assert is_predictive_model_config(cfg)

        cfg = DictConfig({"_target_": "torch.nn.Linear"})
        assert is_predictive_model_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.predictive_models.MyModel"})
        assert is_predictive_model_config(cfg)

    def test_is_predictive_model_config_invalid(self):
        """Test is_predictive_model_config with invalid configs."""
        # Non-model target
        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_predictive_model_config(cfg)

        # Missing _target_
        cfg = DictConfig({"other_field": "value"})
        assert not is_predictive_model_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_predictive_model_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_predictive_model_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_predictive_model_config(cfg)

    def test_is_hooked_transformer_config_valid(self):
        """Test is_hooked_transformer_config with valid HookedTransformer configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformer"})
        assert is_hooked_transformer_config(cfg)

    def test_is_hooked_transformer_config_invalid(self):
        """Test is_hooked_transformer_config with invalid configs."""
        cfg = DictConfig({"_target_": "transformer_lens.HookedTransformerConfig"})
        assert not is_hooked_transformer_config(cfg)

        cfg = DictConfig({"_target_": "torch.nn.Linear"})
        assert not is_hooked_transformer_config(cfg)

        cfg = DictConfig({})
        assert not is_hooked_transformer_config(cfg)

    def test_validate_predictive_model_config_valid(self):
        """Test validate_predictive_model_config with valid configs."""
        # Valid config without name or load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "transformer_lens.HookedTransformer",
                        "cfg": DictConfig(
                            {
                                "_target_": "transformer_lens.HookedTransformerConfig",
                                "n_layers": 2,
                                "d_model": 128,
                                "d_head": 32,
                                "n_ctx": 256,
                                "n_heads": 4,
                                "d_mlp": 512,
                                "act_fn": "relu",
                                "d_vocab": MISSING,
                                "normalization_type": "LN",
                                "device": "cpu",
                                "seed": 42,
                            }
                        ),
                    }
                ),
            }
        )
        validate_predictive_model_config(cfg)

        # Valid config with name and load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "my_model",
                "load_checkpoint_step": 100,
            }
        )
        validate_predictive_model_config(cfg)

        # Valid config with None name and load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": None,
                "load_checkpoint_step": None,
            }
        )
        validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_missing_instance(self):
        """Test validate_predictive_model_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.instance is required"):
            validate_predictive_model_config(cfg)

        cfg = DictConfig({"name": "my_model"})
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.instance is required"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_instance(self):
        """Test validate_predictive_model_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_predictive_model_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
            validate_predictive_model_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_non_model_target(self):
        """Test validate_predictive_model_config raises when instance target is not a model target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.instance._target_ must be a predictive model target"
        ):
            validate_predictive_model_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.instance._target_ must be a predictive model target"
        ):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_name(self):
        """Test validate_predictive_model_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a non-empty string"):
            validate_predictive_model_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a non-empty string"):
            validate_predictive_model_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="PredictiveModelConfig.name must be a string or None"):
            validate_predictive_model_config(cfg)

    def test_validate_predictive_model_config_invalid_load_checkpoint_step(self):
        """Test validate_predictive_model_config raises when load_checkpoint_step is invalid."""
        # Non-integer load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "load_checkpoint_step": "100",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.load_checkpoint_step must be an int or None"
        ):
            validate_predictive_model_config(cfg)

        # Negative load_checkpoint_step
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 5}),
                "load_checkpoint_step": -1,
            }
        )
        with pytest.raises(
            ConfigValidationError, match="PredictiveModelConfig.load_checkpoint_step must be non-negative"
        ):
            validate_predictive_model_config(cfg)


# ============================================================================
# Optimizer Config Tests
# ============================================================================


class TestOptimizerConfig:
    def test_optimizer_config(self):
        """Test creating optimizer config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(OptimizerConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_optimizer_target_valid(self):
        """Test is_optimizer_target with valid optimizer targets."""
        assert is_optimizer_target("torch.optim.Adam")
        assert is_optimizer_target("torch.optim.SGD")
        assert is_optimizer_target("optax.adam")
        assert is_optimizer_target("optax.sgd")

    def test_is_optimizer_target_invalid(self):
        """Test is_optimizer_target with invalid targets."""
        assert not is_optimizer_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_optimizer_target("torch.nn.Linear")
        assert not is_optimizer_target("")

    def test_is_optimizer_config_valid(self):
        """Test is_optimizer_config with valid optimizer configs."""
        cfg = DictConfig({"_target_": "torch.optim.Adam"})
        assert is_optimizer_config(cfg)

        cfg = DictConfig(
            {
                "_target_": "torch.optim.SGD",
                "lr": 0.01,
                "momentum": 0.9,
            }
        )
        assert is_optimizer_config(cfg)

    def test_is_optimizer_config_invalid(self):
        """Test is_optimizer_config with invalid configs."""
        # Non-optimizer target
        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_optimizer_config(cfg)

        # Missing _target_
        cfg = DictConfig({"lr": 0.01})
        assert not is_optimizer_config(cfg)

        # _target_ is None
        cfg = DictConfig({"_target_": None})
        assert not is_optimizer_config(cfg)

        # _target_ is not a string
        cfg = DictConfig({"_target_": 123})
        assert not is_optimizer_config(cfg)

        # Empty config
        cfg = DictConfig({})
        assert not is_optimizer_config(cfg)

    def test_is_pytorch_optimizer_config_valid(self):
        """Test is_pytorch_optimizer_config with valid PyTorch optimizer configs."""
        cfg = DictConfig({"_target_": "torch.optim.Adam"})
        assert is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({"_target_": "torch.optim.SGD"})
        assert is_pytorch_optimizer_config(cfg)

    def test_is_pytorch_optimizer_config_invalid(self):
        """Test is_pytorch_optimizer_config with invalid configs."""
        cfg = DictConfig({"_target_": "optax.adam"})
        assert not is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({})
        assert not is_pytorch_optimizer_config(cfg)

    def test_validate_optimizer_config_valid(self):
        """Test validate_optimizer_config with valid configs."""
        # Valid config without name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.001,
                    }
                ),
            }
        )
        validate_optimizer_config(cfg)  # Should not raise

        # Valid config with name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "optax.adam",
                        "learning_rate": 0.001,
                    }
                ),
                "name": "my_optimizer",
            }
        )
        validate_optimizer_config(cfg)  # Should not raise

        # Valid config with None name
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.SGD",
                        "lr": 0.01,
                        "momentum": 0.9,
                    }
                ),
                "name": None,
            }
        )
        validate_optimizer_config(cfg)  # Should not raise

    def test_validate_optimizer_config_missing_instance(self):
        """Test validate_optimizer_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance is required"):
            validate_optimizer_config(cfg)

        cfg = DictConfig({"name": "my_optimizer"})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance is required"):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_invalid_instance(self):
        """Test validate_optimizer_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"lr": 0.001})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_optimizer_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ cannot be empty or whitespace"):
            validate_optimizer_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_non_optimizer_target(self):
        """Test validate_optimizer_config raises when instance target is not an optimizer target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(
            ConfigValidationError, match="OptimizerConfig.instance._target_ must be an optimizer target"
        ):
            validate_optimizer_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.nn.Linear"})})
        with pytest.raises(
            ConfigValidationError, match="OptimizerConfig.instance._target_ must be an optimizer target"
        ):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_invalid_name(self):
        """Test validate_optimizer_config raises when name is invalid."""
        # Empty string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.optim.Adam", "lr": 0.001}),
                "name": "",
            }
        )
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.name must be a non-empty string"):
            validate_optimizer_config(cfg)

        # Whitespace-only name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.optim.Adam", "lr": 0.001}),
                "name": "   ",
            }
        )
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.name must be a non-empty string"):
            validate_optimizer_config(cfg)

        # Non-string name
        cfg = DictConfig(
            {
                "instance": DictConfig({"_target_": "torch.optim.Adam", "lr": 0.001}),
                "name": 123,
            }
        )
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.name must be a string or None"):
            validate_optimizer_config(cfg)


# ============================================================================
# Base Config Tests
# ============================================================================


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
