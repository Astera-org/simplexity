"""Tests for PersistenceConfig validation.

This module contains tests for persistence configuration validation, including
validation of model persister targets, persister configs, and persistence
configuration instances.
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
from simplexity.structured_configs.persistence import (
    InstanceConfig,
    LocalEquinoxPersisterInstanceConfig,
    LocalPenzaiPersisterInstanceConfig,
    LocalPytorchPersisterInstanceConfig,
    MLFlowPersisterInstanceConfig,
    PersistenceConfig,
    is_model_persister_target,
    is_persister_config,
    update_persister_instance_config,
    validate_local_equinox_persister_instance_config,
    validate_local_penzai_persister_instance_config,
    validate_local_pytorch_persister_instance_config,
    validate_persistence_config,
)


class TestPersistenceConfig:
    """Test PersistenceConfig."""

    def test_persistence_config(self) -> None:
        """Test creating persistence config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(PersistenceConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_model_persister_target_valid(self) -> None:
        """Test is_model_persister_target with valid persister targets."""
        assert is_model_persister_target("simplexity.persistence.mlflow_persister.MLFlowPersister")
        assert is_model_persister_target("simplexity.persistence.local_pytorch_persister.LocalPytorchPersister")

    def test_is_model_persister_target_invalid(self) -> None:
        """Test is_model_persister_target with invalid targets."""
        assert not is_model_persister_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_model_persister_target("torch.optim.Adam")
        assert not is_model_persister_target("")

    def test_is_persister_config_valid(self) -> None:
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

    def test_is_persister_config_invalid(self) -> None:
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

    def test_validate_persistence_config_valid(self) -> None:
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

    def test_validate_persistence_config_missing_instance(self) -> None:
        """Test validate_persistence_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance is required"):
            validate_persistence_config(cfg)

        cfg = DictConfig({"name": "my_persister"})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance is required"):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_invalid_instance(self) -> None:
        """Test validate_persistence_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"other_field": "value"})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_persistence_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_persistence_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_non_persister_target(self) -> None:
        """Test validate_persistence_config raises when instance target is not a persister target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance must be a persister target"):
            validate_persistence_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.optim.Adam"})})
        with pytest.raises(ConfigValidationError, match="PersistenceConfig.instance must be a persister target"):
            validate_persistence_config(cfg)

    def test_validate_persistence_config_invalid_name(self) -> None:
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

    def test_local_persister_configs(self) -> None:
        """Test validation of local persister configs."""
        # Equinox
        eqx_cfg = DictConfig(
            {
                "_target_": "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
                "directory": "/tmp",
                "filename": "model.eqx",
            }
        )
        validate_local_equinox_persister_instance_config(eqx_cfg)
        # Test __init__
        eqx_instance = LocalEquinoxPersisterInstanceConfig(directory="/tmp")
        assert eqx_instance.filename == "model.eqx"
        assert eqx_instance._target_ == "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister"

        # Invalid Equinox filename
        eqx_cfg_invalid = DictConfig(
            {
                "_target_": "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
                "directory": "/tmp",
                "filename": "model.pt",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="LocalEquinoxPersisterInstanceConfig.filename must end with .eqx"
        ):
            validate_local_equinox_persister_instance_config(eqx_cfg_invalid)

        # Penzai
        penzai_cfg = DictConfig(
            {
                "_target_": "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister",
                "directory": "/tmp",
            }
        )
        validate_local_penzai_persister_instance_config(penzai_cfg)
        # Test __init__
        penzai_instance = LocalPenzaiPersisterInstanceConfig(directory="/tmp")
        assert penzai_instance._target_ == "simplexity.persistence.local_penzai_persister.LocalPenzaiPersister"

        # Pytorch
        pt_cfg = DictConfig(
            {
                "_target_": "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister",
                "directory": "/tmp",
                "filename": "model.pt",
            }
        )
        validate_local_pytorch_persister_instance_config(pt_cfg)
        # Test __init__
        pt_instance = LocalPytorchPersisterInstanceConfig(directory="/tmp")
        assert pt_instance.filename == "model.pt"
        assert pt_instance._target_ == "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister"

        # Invalid Pytorch filename
        pt_cfg_invalid = DictConfig(
            {
                "_target_": "simplexity.persistence.local_pytorch_persister.LocalPytorchPersister",
                "directory": "/tmp",
                "filename": "model.eqx",
            }
        )
        with pytest.raises(
            ConfigValidationError, match="LocalPytorchPersisterInstanceConfig.filename must end with .pt"
        ):
            validate_local_pytorch_persister_instance_config(pt_cfg_invalid)

    def test_mlflow_persister_config_init(self) -> None:
        """Test MLFlowPersisterInstanceConfig initialization."""
        config = MLFlowPersisterInstanceConfig(
            experiment_name="test_exp", run_name="test_run", tracking_uri="file:///tmp/mlruns"
        )
        assert config.experiment_name == "test_exp"
        assert config.run_name == "test_run"
        assert config.tracking_uri == "file:///tmp/mlruns"
        assert config._target_ == "simplexity.persistence.mlflow_persister.MLFlowPersister"

    def test_update_persister_instance_config(self) -> None:
        """Test update_persister_instance_config."""
        cfg = OmegaConf.structured(MLFlowPersisterInstanceConfig(experiment_name="old"))
        updated_cfg = DictConfig({"experiment_name": "new", "run_name": "new_run"})

        update_persister_instance_config(cfg, updated_cfg)

        assert cfg.experiment_name == "new"
        assert cfg.run_name == "new_run"
        assert cfg.experiment_id is None
