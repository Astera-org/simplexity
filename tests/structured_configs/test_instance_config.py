"""Tests for InstanceConfig.

This module contains tests for the InstanceConfig dataclass, including
creation from dataclass and derived config classes.
"""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from dataclasses import dataclass

import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.instance import InstanceConfig, validate_instance_config


class TestInstanceConfig:
    """Test InstanceConfig."""

    def test_instance_config(self) -> None:
        """Test creating instance config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(InstanceConfig(_target_="some_target"))
        assert cfg.get("_target_") == "some_target"

    def test_instance_derived_config(self) -> None:
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

    def test_validate_instance_config_valid(self) -> None:
        """Test validate_instance_config with valid configs."""
        # Valid config with _target_
        cfg = DictConfig({"_target_": "some.module.SomeClass"})
        validate_instance_config(cfg)

        # Valid config with expected_target match
        cfg = DictConfig({"_target_": "expected.Target"})
        validate_instance_config(cfg, expected_target="expected.Target")

    def test_validate_instance_config_invalid_target(self) -> None:
        """Test validate_instance_config with invalid _target_."""
        # Missing _target_
        cfg = DictConfig({})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_instance_config(cfg)

        # Empty _target_
        cfg = DictConfig({"_target_": "  "})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a non-empty string"):
            validate_instance_config(cfg)

        # Non-string _target_
        cfg = DictConfig({"_target_": 123})
        with pytest.raises(ConfigValidationError, match="InstanceConfig._target_ must be a string"):
            validate_instance_config(cfg)

    def test_validate_instance_config_unexpected_target(self) -> None:
        """Test validate_instance_config with unexpected target."""
        # Target doesn't match expected
        cfg = DictConfig({"_target_": "actual.Target"})
        with pytest.raises(
            ConfigValidationError, 
            match="InstanceConfig._target_ must be expected.Target, got actual.Target"
        ):
            validate_instance_config(cfg, expected_target="expected.Target")
