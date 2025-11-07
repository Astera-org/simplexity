from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig, OmegaConf

from simplexity.run_management import run_management


def _stub_mlflow_client(monkeypatch: pytest.MonkeyPatch, artifact_source: Path) -> None:
    class DummyClient:
        def __init__(self, tracking_uri: str | None = None):
            self.tracking_uri = tracking_uri

        def get_experiment_by_name(self, name: str):
            return SimpleNamespace(experiment_id="exp-123", name=name)

        def search_runs(self, experiment_ids, filter_string, max_results):
            run_info = SimpleNamespace(run_id="run-123")
            return [SimpleNamespace(info=run_info)]

        def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str) -> str:
            destination = Path(dst_path) / artifact_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(artifact_source, destination)
            return str(destination)

    monkeypatch.setattr(run_management.mlflow, "MlflowClient", DummyClient)


def test_load_config_adds_new_sections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_cfg = OmegaConf.create(
        {
            "predictive_model": {"instance": {"_target_": "transformer_lens.HookedTransformer", "cfg": {"d_model": 4}}},
            "nested": {"sub": {"value": 1}},
        }
    )
    artifact_source = tmp_path / "config.yaml"
    OmegaConf.save(source_cfg, artifact_source)

    _stub_mlflow_client(monkeypatch, artifact_source)

    cfg = OmegaConf.create({"predictive_model": {"instance": {"_target_": "foo"}}})
    load_cfg = DictConfig(
        {
            "tracking_uri": "databricks",
            "experiment_name": "demo",
            "run_name": "reuse",
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


def test_load_config_merges_into_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_cfg = OmegaConf.create({"predictive_model": {"foo": "old", "bar": 1}})
    artifact_source = tmp_path / "config.yaml"
    OmegaConf.save(source_cfg, artifact_source)

    _stub_mlflow_client(monkeypatch, artifact_source)

    cfg = OmegaConf.create({"predictive_model": {"foo": "new"}})
    load_cfg = DictConfig(
        {
            "experiment_name": "demo",
            "run_name": "reuse",
            "configs": {
                "predictive_model": "predictive_model",
            },
        }
    )

    run_management._load_config(cfg, load_cfg)

    assert OmegaConf.select(cfg, "predictive_model.foo") == "new"
    assert OmegaConf.select(cfg, "predictive_model.bar") == 1
