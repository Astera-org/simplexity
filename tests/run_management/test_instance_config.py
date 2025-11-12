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

from omegaconf import DictConfig, OmegaConf

from simplexity.run_management.structured_configs import InstanceConfig


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
