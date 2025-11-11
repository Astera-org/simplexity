from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management import run_management


def _create_run_with_config(
    tmp_path: Path,
    source_cfg: DictConfig,
    *,
    experiment_name: str,
    run_name: str,
) -> tuple[str, str, str]:
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_posix()
    previous_tracking_uri = mlflow.get_tracking_uri()
    run_id: str | None = None
    experiment_id: str | None = None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        experiment_id = experiment.experiment_id
        artifact_file = tmp_path / "config.yaml"
        OmegaConf.save(source_cfg, artifact_file)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_artifact(str(artifact_file))
            run_id = run.info.run_id
    finally:
        if previous_tracking_uri is not None:
            mlflow.set_tracking_uri(previous_tracking_uri)
    assert experiment_id is not None and run_id is not None
    return tracking_uri, experiment_id, run_id


def test_load_config_adds_new_sections(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create(
        {
            "predictive_model": {"instance": {"_target_": "transformer_lens.HookedTransformer", "cfg": {"d_model": 4}}},
            "nested": {"sub": {"value": 1}},
        }
    )
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri, _, _ = _create_run_with_config(
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
    tracking_uri, _, _ = _create_run_with_config(
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


def test_load_config_supports_ids(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create({"predictive_model": {"foo": "id"}})
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri, experiment_id, run_id = _create_run_with_config(
        tmp_path,
        source_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    cfg = OmegaConf.create({})
    load_cfg = DictConfig(
        {
            "tracking_uri": tracking_uri,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "configs": {
                "predictive_model": "copied",
            },
        }
    )

    run_management._load_config(cfg, load_cfg)

    assert OmegaConf.select(cfg, "copied.foo") == "id"


def test_load_config_experiment_name_id_mismatch(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create({"predictive_model": {"foo": "mismatch"}})
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri, experiment_id, run_id = _create_run_with_config(
        tmp_path,
        source_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    cfg = OmegaConf.create({})
    load_cfg = DictConfig(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": "different",
            "experiment_id": experiment_id,
            "run_name": run_name,
            "configs": {"predictive_model": "copied"},
        }
    )
    with pytest.raises(ValueError, match="does not match provided experiment_name"):
        run_management._load_config(cfg, load_cfg)


def test_load_config_run_name_id_mismatch(tmp_path: Path) -> None:
    source_cfg = OmegaConf.create({"predictive_model": {"foo": "mismatch"}})
    experiment_name = "demo"
    run_name = "reuse"
    tracking_uri, experiment_id, run_id = _create_run_with_config(
        tmp_path,
        source_cfg,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    cfg = OmegaConf.create({})
    load_cfg = DictConfig(
        {
            "tracking_uri": tracking_uri,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_name": "different",
            "configs": {"predictive_model": "copied"},
        }
    )
    with pytest.raises(ValueError, match="does not match provided run_name"):
        run_management._load_config(cfg, load_cfg)
