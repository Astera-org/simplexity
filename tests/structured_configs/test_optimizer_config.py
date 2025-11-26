"""Tests for OptimizerConfig validation.

This module contains tests for optimizer configuration validation, including
validation of optimizer targets (PyTorch and Optax), optimizer configs, and
optimizer configuration instances.
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
from simplexity.structured_configs.optimizer import (
    InstanceConfig,
    OptimizerConfig,
    is_optimizer_config,
    is_optimizer_target,
    is_pytorch_optimizer_config,
    validate_optimizer_config,
)


class TestOptimizerConfig:
    """Test OptimizerConfig."""

    def test_optimizer_config(self) -> None:
        """Test creating optimizer config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(OptimizerConfig(instance=InstanceConfig(_target_="some_target")))
        assert OmegaConf.select(cfg, "instance._target_") == "some_target"
        assert cfg.get("name") is None

    def test_is_optimizer_target_valid(self) -> None:
        """Test is_optimizer_target with valid optimizer targets."""
        assert is_optimizer_target("torch.optim.Adam")
        assert is_optimizer_target("torch.optim.SGD")
        assert is_optimizer_target("optax.adam")
        assert is_optimizer_target("optax.sgd")

    def test_is_optimizer_target_invalid(self) -> None:
        """Test is_optimizer_target with invalid targets."""
        assert not is_optimizer_target("simplexity.logging.mlflow_logger.MLFlowLogger")
        assert not is_optimizer_target("torch.nn.Linear")
        assert not is_optimizer_target("")

    def test_is_optimizer_config_valid(self) -> None:
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

    def test_is_optimizer_config_invalid(self) -> None:
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

    def test_is_pytorch_optimizer_config_valid(self) -> None:
        """Test is_pytorch_optimizer_config with valid PyTorch optimizer configs."""
        cfg = DictConfig({"_target_": "torch.optim.Adam"})
        assert is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({"_target_": "torch.optim.SGD"})
        assert is_pytorch_optimizer_config(cfg)

    def test_is_pytorch_optimizer_config_invalid(self) -> None:
        """Test is_pytorch_optimizer_config with invalid configs."""
        cfg = DictConfig({"_target_": "optax.adam"})
        assert not is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})
        assert not is_pytorch_optimizer_config(cfg)

        cfg = DictConfig({})
        assert not is_pytorch_optimizer_config(cfg)

    def test_validate_optimizer_config_valid(self) -> None:
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

    def test_validate_optimizer_config_missing_instance(self) -> None:
        """Test validate_optimizer_config raises when instance is missing."""
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance must be a DictConfig"):
            validate_optimizer_config(cfg)

        cfg = DictConfig({"name": "my_optimizer"})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance must be a DictConfig"):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_invalid_instance(self) -> None:
        """Test validate_optimizer_config raises when instance is invalid."""
        # Instance without _target_
        cfg = DictConfig({"instance": DictConfig({"lr": 0.001})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_optimizer_config(cfg)

        # Instance with empty _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": ""})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_optimizer_config(cfg)

        # Instance with non-string _target_
        cfg = DictConfig({"instance": DictConfig({"_target_": 123})})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_non_optimizer_target(self) -> None:
        """Test validate_optimizer_config raises when instance target is not an optimizer target."""
        cfg = DictConfig({"instance": DictConfig({"_target_": "simplexity.logging.mlflow_logger.MLFlowLogger"})})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance must be an optimizer target"):
            validate_optimizer_config(cfg)

        cfg = DictConfig({"instance": DictConfig({"_target_": "torch.nn.Linear"})})
        with pytest.raises(ConfigValidationError, match="OptimizerConfig.instance must be an optimizer target"):
            validate_optimizer_config(cfg)

    def test_validate_optimizer_config_invalid_name(self) -> None:
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

    def test_validate_pytorch_adam_instance_config(self) -> None:
        """Test validation of AdamInstanceConfig."""
        # Valid config
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.001,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                        "amsgrad": False,
                    }
                )
            }
        )
        validate_optimizer_config(cfg)

        # Invalid lr
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.Adam",
                        "lr": -0.001,
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="AdamInstanceConfig.lr must be positive"):
            validate_optimizer_config(cfg)

        # Invalid betas length
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.Adam",
                        "betas": [0.9],
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="AdamInstanceConfig.betas must have length 2"):
            validate_optimizer_config(cfg)

        # Invalid betas values
        cfg = DictConfig(
            {
                "instance": DictConfig(
                    {
                        "_target_": "torch.optim.Adam",
                        "betas": [-0.9, 0.999],
                    }
                )
            }
        )
        with pytest.raises(ConfigValidationError, match="AdamInstanceConfig.betas\\[0\\] must be non-negative"):
            validate_optimizer_config(cfg)
