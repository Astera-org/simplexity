"""Test MLflow defaults."""

from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.structured_configs.mlflow_defaults import load_mlflow_defaults


@pytest.fixture(autouse=True)
def _patch_mlflow_helpers(mocker):
    """Automatically patch common MLflow helper functions for all tests."""
    mocker.patch("simplexity.structured_configs.mlflow_defaults.validate_mlflow_config")
    mocker.patch("simplexity.structured_configs.mlflow_defaults.resolve_mlflow_config")


@pytest.fixture
def base_cfg() -> DictConfig:
    """Set up the base config."""
    return OmegaConf.create(
        {
            "previous_run": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id",
            },
            "other_section": {"foo": "bar"},
        }
    )


@pytest.fixture
def mock_download(mocker) -> MagicMock:
    """Create a mock MLflow client instance."""
    mock_client_instance = MagicMock(spec=MlflowClient)
    mocker.patch(
        "simplexity.structured_configs.mlflow_defaults.MlflowClient",
        return_value=mock_client_instance,
    )
    return mock_client_instance.download_artifacts


def test_load_default_config_at_root(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading basic config."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("loaded_key: loaded_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    assert loaded_cfg.get("loaded_key") == "loaded_value"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_load_default_config_at_package(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading at package."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("loaded_key: loaded_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run@dest"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "dest.loaded_key") == "loaded_value"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"
    assert mock_download.call_args.kwargs["path"] == "config.yaml"


def test_load_nondefault_config(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading basic config."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("loaded_key: loaded_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: nondefault_config#"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    assert loaded_cfg.get("loaded_key") == "loaded_value"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"
    assert mock_download.call_args.kwargs["path"] == "nondefault_config.yaml"


def test_load_subconfig(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading subconfig."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("root:\n  sub:\n    nested: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: root.sub"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "sub.nested") == "value"
    assert mock_download.call_args.kwargs["path"] == "config.yaml"


def test_load_subconfig_select(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading subconfig select."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("sub:\n  nested: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run@dest: nondefault_config#sub.nested"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert loaded_cfg.get("dest") == "value"


def test_implicit_config_select(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test implicit artifact="config" when option has no / or #."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("sub: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run@dest: sub"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert loaded_cfg.get("dest") == "value"
    mock_download.assert_called_with(run_id="test_run_id", path="config.yaml", dst_path=ANY)


def test_implicit_artifact_select(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test implicit select=root when option has /."""
    artifact_path = tmp_path / "artifact.yaml"
    artifact_path.write_text("key: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run@dest: path/to/artifact"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "dest.key") == "value"
    mock_download.assert_called_with(run_id="test_run_id", path="path/to/artifact.yaml", dst_path=ANY)


def test_override_flag(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test override flag replaces instead of merging."""
    # Artifact contains different keys than base config to verify replacement
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("new_key: new_value\nother_key: other_value\n")
    mock_download.return_value = str(artifact_path)

    # Place _self_ first so original config is merged, then override replaces it
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["_self_", "override previous_run@other_section: nondefault_config#"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    # Verify override replaced the entire other_section (no merge)
    assert OmegaConf.select(loaded_cfg, "other_section.new_key") == "new_value"
    assert OmegaConf.select(loaded_cfg, "other_section.other_key") == "other_value"
    # Original foo: bar should be gone (replaced, not merged)
    assert OmegaConf.select(loaded_cfg, "other_section.foo") is None


def test_override_flag_at_root(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test override flag at root level with explicit _self_ first."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("root_key: root_value\nother_key: other_value\n")
    mock_download.return_value = str(artifact_path)

    # To override everything, explicitly include _self_ first, then override
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["_self_", "override previous_run"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    # Verify override replaced entire config at root (after _self_ was processed)
    assert OmegaConf.select(loaded_cfg, "root_key") == "root_value"
    assert OmegaConf.select(loaded_cfg, "other_key") == "other_value"
    # Original keys should be gone (override replaced everything)
    assert OmegaConf.select(loaded_cfg, "other_section") is None
    assert OmegaConf.select(loaded_cfg, "previous_run") is None


def test_override_flag_at_root_without_explicit_self(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test override flag at root level without explicit _self_ - _self_ is auto-appended."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("root_key: root_value\nother_key: other_value\n")
    mock_download.return_value = str(artifact_path)

    # Without explicit _self_, it will be auto-appended at the end
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["override previous_run"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    # Override replaced everything, but _self_ was auto-appended and merged after
    # So original keys from _self_ should be back (last entry wins for conflicting keys)
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"
    assert OmegaConf.select(loaded_cfg, "previous_run.tracking_uri") == "databricks"
    # Keys from override that don't conflict with _self_ should still be present
    assert OmegaConf.select(loaded_cfg, "root_key") == "root_value"
    assert OmegaConf.select(loaded_cfg, "other_key") == "other_value"


def test_override_flag_nested_path(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test override flag with nested package path."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("nested_key: nested_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["_self_", "override previous_run@nested.path: config#"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    # Verify override replaced at nested path
    assert OmegaConf.select(loaded_cfg, "nested.path.nested_key") == "nested_value"
    # Original other_section should still exist (not affected)
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_override_with_explicit_self(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test that _self_ is processed when explicitly included after override."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("key1: value1\nkey2: value2\n")
    mock_download.return_value = str(artifact_path)

    # Test with override followed by explicit _self_
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": [
                "override previous_run@other_section: config#",
                "_self_",  # Explicitly included, should merge after override
            ],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)

    # Override replaced other_section, but _self_ merged after, so original foo should be back
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"  # From _self_
    assert OmegaConf.select(loaded_cfg, "other_section.key1") == "value1"  # From override
    assert OmegaConf.select(loaded_cfg, "other_section.key2") == "value2"  # From override


def test_override_vs_merge_behavior(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test that override replaces while normal entry merges."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("key1: value1\nkey2: value2\n")
    mock_download.return_value = str(artifact_path)

    # Test with merge (no override)
    cfg_merge = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["_self_", "previous_run@other_section: config#"],
        },
    )
    loaded_merge: DictConfig = load_mlflow_defaults(cfg_merge)
    # With merge, original foo should be preserved
    assert OmegaConf.select(loaded_merge, "other_section.foo") == "bar"
    assert OmegaConf.select(loaded_merge, "other_section.key1") == "value1"
    assert OmegaConf.select(loaded_merge, "other_section.key2") == "value2"

    # Test with override
    cfg_override = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["_self_", "override previous_run@other_section: config#"],
        },
    )
    loaded_override: DictConfig = load_mlflow_defaults(cfg_override)
    # With override, original foo should be replaced
    assert OmegaConf.select(loaded_override, "other_section.foo") is None
    assert OmegaConf.select(loaded_override, "other_section.key1") == "value1"
    assert OmegaConf.select(loaded_override, "other_section.key2") == "value2"


def test_optional_flag_missing_artifact(base_cfg: DictConfig):
    """Test optional flag with missing artifact."""

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run: null"],
        },
    )
    # Should not raise
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert loaded_cfg == base_cfg


def test_null_option_error(base_cfg: DictConfig):
    """Test null option raises error."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: null"],
        },
    )
    # null option returns just the target config in new logic unless specific logic handles it differently
    # Actually user logic for option="null" returns _ParsedEntry with artifact_path=None
    # And _get_target_config handles artifact_path=None by warning and returning None
    # So "mlflow: null" should be valid but yield nothing in accumulator if not trapped earlier?
    # Re-reading logic in Step 245:
    # _parse_entry returns artifact_path=None if option="null"
    # _get_target_config checks if parsed_entry.artifact_path is None: logs Warning, returns None.
    # _process_entry sees loaded_config is None.
    # It raises ValueError("Target config not found for entry...") if not optional.
    # So it SHOULD raise ValueError.

    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_null_option_optional(base_cfg: DictConfig):
    """Test null option with optional flag."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run: null"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_null_option_dict_syntax_error(base_cfg: DictConfig):
    """Test null option with YAML dict syntax raises error."""
    # YAML dict syntax: - previous_run: null becomes DictConfig with None value
    # This should be treated the same as string format: "previous_run: null"
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": [OmegaConf.create({"previous_run": None})],
        },
    )
    # Should raise ValueError because null option without optional flag is mandatory
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_composition_order(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test _self_ placement in composition order."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("foo: mlflow_value\n")
    mock_download.return_value = str(artifact_path)

    # Case 1: _self_ (default/original) before mlflow -> mlflow overrides
    cfg1 = OmegaConf.merge(base_cfg, {"mlflow_defaults": ["_self_", "previous_run@other_section: nondefault_config#"]})
    loaded1: DictConfig = load_mlflow_defaults(cfg1)
    assert OmegaConf.select(loaded1, "other_section.foo") == "mlflow_value"

    # Case 2: mlflow before _self_ -> original overrides (last entry wins)
    cfg2 = OmegaConf.merge(base_cfg, {"mlflow_defaults": ["previous_run@other_section: nondefault_config#", "_self_"]})
    loaded2: DictConfig = load_mlflow_defaults(cfg2)
    assert OmegaConf.select(loaded2, "other_section.foo") == "bar"  # Original value wins (last entry)


def test_config_entry_syntax(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test CONFIG syntax: TARGET (no option)."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("key: val\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "key") == "val"
    # Implicitly loads "config" artifact
    mock_download.assert_called_with(run_id="test_run_id", path="config.yaml", dst_path=ANY)


def test_missing_target_error(base_cfg: DictConfig):
    """Test missing target raises error."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional : option"],
        },
    )
    with pytest.raises(ValueError, match="Invalid MLflow default entry"):
        load_mlflow_defaults(cfg)


def test_invalid_entry_format(base_cfg: DictConfig):
    """Test invalid entry format raises error (line 64)."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["invalid@format@too@many@ats"],
        },
    )
    with pytest.raises(ValueError, match="Invalid MLflow default entry"):
        load_mlflow_defaults(cfg)


def test_missing_target_node(base_cfg: DictConfig):
    """Test missing target node in config (lines 96-97)."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["nonexistent: config"],
        },
    )
    # Should raise ValueError because target node not found and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_missing_target_node_optional(base_cfg: DictConfig):
    """Test missing target node with optional flag."""
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional nonexistent: config"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_validate_mlflow_config_error(base_cfg: DictConfig, mocker):
    """Test ConfigValidationError from validate_mlflow_config (lines 101-103)."""
    # Make validate_mlflow_config raise ConfigValidationError
    mock_validate = mocker.patch("simplexity.structured_configs.mlflow_defaults.validate_mlflow_config")
    mock_validate.side_effect = ConfigValidationError("Invalid config")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: config"],
        },
    )
    # Should raise ValueError because validation failed and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_validate_mlflow_config_error_optional(base_cfg: DictConfig, mocker):
    """Test ConfigValidationError with optional flag."""
    # Make validate_mlflow_config raise ConfigValidationError
    mock_validate = mocker.patch("simplexity.structured_configs.mlflow_defaults.validate_mlflow_config")
    mock_validate.side_effect = ConfigValidationError("Invalid config")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run: config"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_resolve_mlflow_config_error(base_cfg: DictConfig, mocker):
    """Test ValueError from resolve_mlflow_config (lines 107-109)."""
    # Make resolve_mlflow_config raise ValueError
    mock_resolve = mocker.patch("simplexity.structured_configs.mlflow_defaults.resolve_mlflow_config")
    mock_resolve.side_effect = ValueError("Resolution failed")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: config"],
        },
    )
    # Should raise ValueError because resolution failed and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_resolve_mlflow_config_error_optional(base_cfg: DictConfig, mocker):
    """Test ValueError from resolve_mlflow_config with optional flag."""
    # Make resolve_mlflow_config raise ValueError
    mock_resolve = mocker.patch("simplexity.structured_configs.mlflow_defaults.resolve_mlflow_config")
    mock_resolve.side_effect = ValueError("Resolution failed")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional mlflow: config"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_omegaconf_load_error(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test Exception from OmegaConf.load (lines 124-126)."""
    # Create a file with invalid YAML content
    artifact_path = tmp_path / "invalid.yaml"
    artifact_path.write_text("invalid: yaml: content: [unclosed")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["mlflow: config"],
        },
    )
    # Should raise ValueError because load failed and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_omegaconf_load_error_optional(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test Exception from OmegaConf.load with optional flag."""
    # Create a file with invalid YAML content
    artifact_path = tmp_path / "invalid.yaml"
    artifact_path.write_text("invalid: yaml: content: [unclosed")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run: config"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_select_path_not_found(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test select path not found in artifact (lines 133-136)."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("key: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run@dest: nondefault_config#nonexistent.path"],
        },
    )
    # Should raise ValueError because select path not found and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_select_path_not_found_optional(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test select path not found with optional flag."""
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("key: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run@dest: nondefault_config#nonexistent.path"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_non_dictconfig_error(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test non-DictConfig error when not optional (lines 157-159)."""
    # Create artifact with list (non-dict) root value
    # Use "nondefault_config#" to avoid select_path, so we get the full ListConfig
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("- item1\n- item2\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: nondefault_config#"],
        },
    )
    # Should raise ValueError because loaded_config is ListConfig (not DictConfig) and not optional
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_non_dictconfig_optional(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test non-DictConfig with optional flag."""
    # Create artifact with list (non-dict) root value
    # Use "nondefault_config#" to avoid select_path, so we get the full ListConfig
    artifact_path = tmp_path / "nondefault_config.yaml"
    artifact_path.write_text("- item1\n- item2\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run: nondefault_config#"],
        },
    )
    # Should not raise, just return accumulator
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section") == cfg.other_section


def test_no_mlflow_defaults_key():
    """Test early return when mlflow_defaults key is missing (line 172)."""
    # Config without mlflow_defaults key
    cfg = OmegaConf.create(
        {
            "previous_run": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id",
            },
            "other_section": {"foo": "bar"},
        }
    )
    # Should return cfg unchanged
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert loaded_cfg == cfg
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_multiple_entries_different_runs(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading configs from multiple different runs."""
    # Setup two different runs with different configs
    artifact_path_1 = tmp_path / "config_run1.yaml"
    run_1 = DictConfig({"run1_key": "run1_value", "run1_section": {"nested": "run1_nested"}})
    artifact_path_1.write_text(OmegaConf.to_yaml(run_1))
    artifact_path_2 = tmp_path / "config_run2.yaml"
    run_2 = DictConfig({"run2_key": "run2_value", "run2_section": {"nested": "run2_nested"}})
    artifact_path_2.write_text(OmegaConf.to_yaml(run_2))

    # Mock download_artifacts to return different paths for different run_ids
    def download_side_effect(run_id=None, _path=None, _dst_path=None, **_kwargs):
        if run_id == "test_run_id_1":
            return str(artifact_path_1)
        if run_id == "test_run_id_2":
            return str(artifact_path_2)
        return ""

    mock_download.side_effect = download_side_effect

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "run1": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id_1",
            },
            "run2": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id_2",
            },
            "mlflow_defaults": ["run1@model1", "run2@model2"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "model1") == run_1
    assert OmegaConf.select(loaded_cfg, "model2") == run_2
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_multiple_entries_same_run(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test loading multiple artifacts from the same run."""
    # Setup multiple artifacts from the same run
    artifact_path_1 = tmp_path / "config.yaml"
    artifact_path_1.write_text("artifact1_key: artifact1_value\n")
    artifact_path_2 = tmp_path / "other_artifact.yaml"
    artifact_path_2.write_text("artifact2_key: artifact2_value\n")

    # Mock download_artifacts to return different paths based on artifact path
    def download_side_effect(_run_id=None, path=None, _dst_path=None, **_kwargs):
        # Path might be "config" or "config.yaml" depending on parsing
        if path in ("config", "config.yaml"):
            return str(artifact_path_1)
        if path in ("other_artifact", "other_artifact.yaml"):
            return str(artifact_path_2)
        return ""

    mock_download.side_effect = download_side_effect

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": [
                "previous_run@section1: config#",
                "previous_run@section2: other_artifact#",
            ],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "section1.artifact1_key") == "artifact1_value"
    assert OmegaConf.select(loaded_cfg, "section2.artifact2_key") == "artifact2_value"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_multiple_entries_shared_keys_last_wins(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test that last entry wins when keys conflict."""
    # Setup entries that load configs with overlapping keys
    artifact_path_1 = tmp_path / "config1.yaml"
    artifact_path_1.write_text("shared_key: first_value\nunique1: value1\n")
    artifact_path_2 = tmp_path / "config2.yaml"
    artifact_path_2.write_text("shared_key: second_value\nunique2: value2\n")

    def download_side_effect(_run_id=None, path=None, _dst_path=None, **_kwargs):
        # Path might be "config1" or "config1.yaml" depending on parsing
        if path in ("config1", "config1.yaml"):
            return str(artifact_path_1)
        if path in ("config2", "config2.yaml"):
            return str(artifact_path_2)
        return ""

    mock_download.side_effect = download_side_effect

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "run1": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id_1",
            },
            "run2": {
                "tracking_uri": "databricks",
                "run_id": "test_run_id_2",
            },
            "mlflow_defaults": [
                "run1: config1#",
                "run2: config2#",
            ],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    # Last entry should win for shared_key
    assert OmegaConf.select(loaded_cfg, "shared_key") == "second_value"
    # Both unique keys should be present
    assert OmegaConf.select(loaded_cfg, "unique1") == "value1"
    assert OmegaConf.select(loaded_cfg, "unique2") == "value2"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_option_trailing_hash(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test option with trailing # (artifact path only)."""
    # Test: "custom#" should load artifact "custom" at root
    artifact_path = tmp_path / "custom.yaml"
    artifact_path.write_text("key: value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: custom#"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "key") == "value"
    assert mock_download.call_args.kwargs["path"] == "custom.yaml"


def test_option_hash_empty_select(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test option with # but empty select path."""
    # Test: "artifact#" should load artifact "artifact" at root
    artifact_path = tmp_path / "artifact.yaml"
    artifact_path.write_text("root_key: root_value\nnested:\n  nested_key: nested_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run: artifact#"],
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    # Should load entire artifact at root
    assert OmegaConf.select(loaded_cfg, "root_key") == "root_value"
    assert OmegaConf.select(loaded_cfg, "nested.nested_key") == "nested_value"
    assert mock_download.call_args.kwargs["path"] == "artifact.yaml"


def test_mlflow_exception_caught_by_download(base_cfg: DictConfig, mock_download: MagicMock):
    """Test that MlflowException from download_artifacts is caught."""
    mock_download.side_effect = MlflowException("Artifact not found on remote store")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run"],
        },
    )
    with pytest.raises(ValueError, match="Target config not found for entry"):
        load_mlflow_defaults(cfg)


def test_mlflow_exception_caught_optional(base_cfg: DictConfig, mock_download: MagicMock):
    """Test that MlflowException from download_artifacts is caught with optional flag."""
    mock_download.side_effect = MlflowException("Artifact not found on remote store")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["optional previous_run"],
        },
    )
    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_string_mlflow_defaults(base_cfg: DictConfig, mock_download: MagicMock, tmp_path: Path):
    """Test that a string mlflow_defaults value is handled correctly."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("loaded_key: loaded_value\n")
    mock_download.return_value = str(artifact_path)

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": "previous_run",
        },
    )

    loaded_cfg: DictConfig = load_mlflow_defaults(cfg)
    assert loaded_cfg.get("loaded_key") == "loaded_value"
    assert OmegaConf.select(loaded_cfg, "other_section.foo") == "bar"


def test_resolve_mlflow_config_called_with_create_if_missing_false(
    base_cfg: DictConfig,
    mock_download: MagicMock,
    tmp_path: Path,
    mocker,
):
    """Test that resolve_mlflow_config is called with create_if_missing=False."""
    artifact_path = tmp_path / "config.yaml"
    artifact_path.write_text("loaded_key: loaded_value\n")
    mock_download.return_value = str(artifact_path)

    mock_resolve = mocker.patch("simplexity.structured_configs.mlflow_defaults.resolve_mlflow_config")

    cfg = OmegaConf.merge(
        base_cfg,
        {
            "mlflow_defaults": ["previous_run"],
        },
    )

    load_mlflow_defaults(cfg)
    mock_resolve.assert_called()
    for c in mock_resolve.call_args_list:
        assert c.kwargs.get("create_if_missing") is False
