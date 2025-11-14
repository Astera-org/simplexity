from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management.components import Components
from simplexity.run_management.mlflow_loaders import load_model_and_generative_process, load_run_components


def _create_run_with_config(
    tmp_path: Path,
    config_dict: dict,
    *,
    experiment_name: str,
    run_name: str,
) -> tuple[str, str]:
    """Create an MLflow run with a config artifact."""
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_posix()
    previous_tracking_uri = mlflow.get_tracking_uri()
    run_id: str | None = None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        artifact_file = tmp_path / "config.yaml"
        OmegaConf.save(OmegaConf.create(config_dict), artifact_file)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_artifact(str(artifact_file))
            run_id = run.info.run_id
    finally:
        if previous_tracking_uri is not None:
            mlflow.set_tracking_uri(previous_tracking_uri)
    assert run_id is not None
    return tracking_uri, run_id


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_run_components_basic(mock_setup, tmp_path: Path) -> None:
    """Test basic loading of components from an MLflow run (without persistence)."""
    config = {
        "predictive_model": {
            "instance": {
                "_target_": "some.model.Model",
                "param": 123,
            },
        },
        "generative_process": {
            "instance": {
                "_target_": "some.gp.GP",
                "param": 456,
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_components = Components()
    mock_components.predictive_models = {"predictive_model.instance": Mock()}
    mock_components.generative_processes = {"generative_process.instance": Mock()}
    mock_setup.return_value = mock_components

    components = load_run_components(run_id=run_id, tracking_uri=tracking_uri)

    assert components is not None
    assert mock_setup.called

    called_cfg: DictConfig = mock_setup.call_args[0][0]
    assert OmegaConf.select(called_cfg, "predictive_model.instance._target_") == "some.model.Model"
    assert OmegaConf.select(called_cfg, "predictive_model.instance.param") == 123
    assert OmegaConf.select(called_cfg, "generative_process.instance._target_") == "some.gp.GP"
    assert OmegaConf.select(called_cfg, "generative_process.instance.param") == 456
    # Verify persistence was not loaded by default
    assert OmegaConf.select(called_cfg, "persistence") is None


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_run_components_with_custom_config_keys(mock_setup, tmp_path: Path) -> None:
    """Test loading components with custom config key mapping."""
    config = {
        "old_model": {
            "instance": {
                "_target_": "some.model.Model",
                "param": 789,
            },
        },
        "old_gp": {
            "instance": {
                "_target_": "some.gp.GP",
                "param": 101112,
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_components = Components()
    mock_setup.return_value = mock_components

    components = load_run_components(
        run_id=run_id,
        tracking_uri=tracking_uri,
        config_keys={
            "old_model": "predictive_model",
            "old_gp": "generative_process",
        },
    )

    assert components is not None
    assert mock_setup.called

    called_cfg: DictConfig = mock_setup.call_args[0][0]
    assert OmegaConf.select(called_cfg, "predictive_model.instance._target_") == "some.model.Model"
    assert OmegaConf.select(called_cfg, "predictive_model.instance.param") == 789
    assert OmegaConf.select(called_cfg, "generative_process.instance._target_") == "some.gp.GP"
    assert OmegaConf.select(called_cfg, "generative_process.instance.param") == 101112


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_run_components_with_persistence(mock_setup, tmp_path: Path) -> None:
    """Test loading components including persistence when explicitly requested."""
    config = {
        "predictive_model": {
            "instance": {
                "_target_": "some.model.Model",
            },
        },
        "generative_process": {
            "instance": {
                "_target_": "some.gp.GP",
            },
        },
        "persistence": {
            "instance": {
                "_target_": "some.persister.Persister",
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_components = Components()
    mock_setup.return_value = mock_components

    # Explicitly request persistence
    components = load_run_components(
        run_id=run_id,
        tracking_uri=tracking_uri,
        config_keys={
            "predictive_model": "predictive_model",
            "generative_process": "generative_process",
            "persistence": "persistence",
        },
    )

    assert components is not None

    called_cfg: DictConfig = mock_setup.call_args[0][0]
    assert OmegaConf.select(called_cfg, "predictive_model.instance._target_") is not None
    assert OmegaConf.select(called_cfg, "generative_process.instance._target_") is not None
    assert OmegaConf.select(called_cfg, "persistence.instance._target_") is not None


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_model_and_generative_process(mock_setup, tmp_path: Path) -> None:
    """Test convenience function for loading model and generative process."""
    config = {
        "predictive_model": {
            "instance": {
                "_target_": "some.model.Model",
            },
        },
        "generative_process": {
            "instance": {
                "_target_": "some.gp.GP",
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_model = Mock()
    mock_gp = Mock()
    mock_components = Components()
    mock_components.predictive_models = {"predictive_model.instance": mock_model}
    mock_components.generative_processes = {"generative_process.instance": mock_gp}
    mock_setup.return_value = mock_components

    model, gp = load_model_and_generative_process(run_id=run_id, tracking_uri=tracking_uri)

    assert model is mock_model
    assert gp is mock_gp


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_model_and_generative_process_missing_model(mock_setup, tmp_path: Path) -> None:
    """Test error handling when model is missing from config."""
    config = {
        "generative_process": {
            "instance": {
                "_target_": "some.gp.GP",
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_components = Components()
    mock_components.generative_processes = {"generative_process.instance": Mock()}
    mock_setup.return_value = mock_components

    with pytest.raises(KeyError, match="Config key 'predictive_model' not found"):
        load_model_and_generative_process(run_id=run_id, tracking_uri=tracking_uri)


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_model_and_generative_process_missing_gp(mock_setup, tmp_path: Path) -> None:
    """Test error handling when generative process is missing from config."""
    config = {
        "predictive_model": {
            "instance": {
                "_target_": "some.model.Model",
            },
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    mock_components = Components()
    mock_components.predictive_models = {"predictive_model.instance": Mock()}
    mock_setup.return_value = mock_components

    with pytest.raises(KeyError, match="Config key 'generative_process' not found"):
        load_model_and_generative_process(run_id=run_id, tracking_uri=tracking_uri)


def test_load_run_components_invalid_run_id(tmp_path: Path) -> None:
    """Test error handling for invalid run ID."""
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_posix()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_experiment")

    with pytest.raises(ValueError, match="Run with id 'nonexistent_run_id' not found"):
        load_run_components(run_id="nonexistent_run_id", tracking_uri=tracking_uri)


@patch("simplexity.run_management.mlflow_loaders._setup")
def test_load_run_components_missing_default_keys(mock_setup, tmp_path: Path) -> None:
    """Test error handling when default config keys are not found in source."""
    config = {
        "some_other_config": {
            "value": 123,
        },
    }

    tracking_uri, run_id = _create_run_with_config(
        tmp_path,
        config,
        experiment_name="test_experiment",
        run_name="test_run",
    )

    # With default config_keys, should raise KeyError for missing 'predictive_model'
    with pytest.raises(KeyError, match="Config key 'predictive_model' not found"):
        load_run_components(run_id=run_id, tracking_uri=tracking_uri)
