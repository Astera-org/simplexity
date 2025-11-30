"""End-to-end training tests."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from pathlib import Path
from typing import cast

import mlflow
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.end_to_end.training import CONFIG_DIR, CONFIG_NAME, TrainingRunConfig, train


def test_training(tmp_path: Path) -> None:
    """Test training."""
    mlflow_db = tmp_path / "mlflow.db"
    mlflow_uri = f"sqlite:///{mlflow_db.absolute()}"
    overrides = [
        f"mlflow.tracking_uri={mlflow_uri}",
        f"mlflow.registry_uri={mlflow_uri}",
    ]
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    train(cfg)

    cfg = cast(TrainingRunConfig, cfg)

    client = mlflow.MlflowClient(tracking_uri=mlflow_uri, registry_uri=mlflow_uri)
    run_id = cfg.mlflow.run_id
    assert run_id is not None
    run = client.get_run(run_id)
    assert run is not None

    # Tags
    tags = run.data.tags
    for key, value in cfg.tags.items():
        assert tags[key] == value
    assert "strict" in tags
    for key in ["commit", "commit_full", "dirty", "branch", "remote"]:
        assert f"git.main.{key}" in tags

    # Parameters
    parameters = run.data.params
    assert len(parameters) > 0

    # Config
    config_path = client.download_artifacts(
        run_id,
        path="config.yaml",
        dst_path=str(tmp_path / "config.yaml"),
    )
    assert OmegaConf.load(config_path) == cfg

    # Metrics
    def get_metric_values(metric_name: str) -> np.ndarray:
        metric_history = client.get_metric_history(run.info.run_id, metric_name)
        return np.array([metric.value for metric in metric_history])

    train_loss = get_metric_values("train/loss")
    eval_loss = get_metric_values("eval/loss")
    assert train_loss.shape == (cfg.training.num_steps // cfg.training.log_every + 1,)
    assert eval_loss.shape == (cfg.training.num_steps // cfg.training.evaluate_every + 1,)
    assert np.all(train_loss > 0)
    assert np.all(eval_loss > 0)

    # Checkpoints
    model_dir = cfg.persistence.instance.model_dir or "models"  # type: ignore
    checkpoints = client.list_artifacts(run.info.run_id, model_dir)
    assert len(checkpoints) == cfg.training.num_steps // cfg.training.checkpoint_every + 1

    # Logged models
    registered_model_name = cfg.predictive_model.name or "test_model"
    model = client.get_registered_model(registered_model_name)
    assert model is not None
