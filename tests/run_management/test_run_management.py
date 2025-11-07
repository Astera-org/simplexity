from __future__ import annotations

from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management import run_management


def _create_run_with_config(
    tmp_path: Path,
    source_cfg: DictConfig,
    *,
    experiment_name: str,
    run_name: str,
) -> str:
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_posix()
    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        artifact_file = tmp_path / "config.yaml"
        OmegaConf.save(source_cfg, artifact_file)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact(str(artifact_file))
    finally:
        if previous_tracking_uri is not None:
            mlflow.set_tracking_uri(previous_tracking_uri)
    return tracking_uri


def test_load_config_adds_new_sections(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create(
        {
            "predictive_model": {"instance": {"_target_": "transformer_lens.HookedTransformer", "cfg": {"d_model": 4}}},
            "nested": {"sub": {"value": 1}},
        }
    )
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri = _create_run_with_config(
        tmp_path,
        source_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    cfg = OmegaConf.create({"predictive_model": {"instance": {"_target_": "foo"}}})
    load_cfg = DictConfig(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "configs": {
                "predictive_model": "old_models.model_1",
                "nested.sub": "copied.sub",
            },
        }
    )

    run_management._load_config(cfg, load_cfg)

    assert OmegaConf.select(cfg, "old_models.model_1.instance.cfg.d_model") == 4
    assert OmegaConf.select(cfg, "copied.sub.value") == 1
    # Ensure existing sections remain untouched
    assert OmegaConf.select(cfg, "predictive_model.instance._target_") == "foo"


def test_load_config_merges_into_existing(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create({"predictive_model": {"foo": "old", "bar": 1}})
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri = _create_run_with_config(
        tmp_path,
        source_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    cfg = OmegaConf.create({"predictive_model": {"foo": "new"}})
    load_cfg = DictConfig(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "configs": {
                "predictive_model": "predictive_model",
            },
        }
    )

    run_management._load_config(cfg, load_cfg)

    assert OmegaConf.select(cfg, "predictive_model.foo") == "new"
    assert OmegaConf.select(cfg, "predictive_model.bar") == 1
