"""Tests for MLFlowConfig validation.

This module contains tests for MLFlow configuration validation, including
validation of experiment_name, run_name, tracking_uri, registry_uri, and
downgrade_unity_catalog fields.
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
from simplexity.structured_configs.mlflow import MLFlowConfig, validate_mlflow_config


class TestMLFlowConfig:
    """Test MLFlowConfig."""

    def test_mlflow_config(self) -> None:
        """Test creating mlflow config from dataclass."""
        cfg: DictConfig = OmegaConf.structured(MLFlowConfig(experiment_name="some_experiment", run_name="some_run"))
        assert cfg.get("experiment_name") == "some_experiment"
        assert cfg.get("run_name") == "some_run"
        assert cfg.get("tracking_uri") is None
        assert cfg.get("registry_uri") is None
        assert cfg.get("downgrade_unity_catalog") is None

    def test_validate_mlflow_config_valid(self) -> None:
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

    def test_validate_mlflow_config_missing_required_fields(self) -> None:
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

    def test_validate_mlflow_config_invalid_downgrade_unity_catalog(self) -> None:
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
    def test_validate_mlflow_config_invalid_uri(self, uri_type: str) -> None:
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
