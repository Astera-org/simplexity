from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from simplexity.loaders.experiment_loader import ExperimentLoader
from simplexity.logging.run_reader import RunReader


class _StubReader(RunReader):
    def __init__(self, cfg):
        self._cfg = cfg

    def get_config(self):
        return self._cfg

    def get_params(self):  # pragma: no cover - not used
        return {}

    def get_tags(self):  # pragma: no cover - not used
        return {}

    def get_metrics(self, pattern: str | None = None):  # pragma: no cover - not used
        return pd.DataFrame(columns=["metric", "step", "value", "timestamp"])

    def list_artifacts(self, path: str | None = None):  # pragma: no cover - not used
        return []

    def download_artifact(self, path: str, dst: str | Path | None = None):  # pragma: no cover - not used
        raise NotImplementedError


def test_list_checkpoints_without_model_instantiation(tmp_path: Path):
    # Prepare a config that would fail if model instantiation is attempted
    cfg_dict = {
        "predictive_model": {
            "name": "broken",
            "instance": {
                "_target_": "nonexistent.module.Class",  # Must not be imported in this test
            },
        },
        "persistence": {
            "name": "local_equinox_persister",
            "instance": {
                "_target_": "simplexity.persistence.local_equinox_persister.LocalEquinoxPersister",
                "directory": str(tmp_path),
                "filename": "model.eqx",
            },
        },
    }
    cfg = OmegaConf.create(cfg_dict)

    # Create fake checkpoints
    (tmp_path / "0").mkdir(parents=True, exist_ok=True)
    (tmp_path / "0" / "model.eqx").write_bytes(b"")
    (tmp_path / "3").mkdir(parents=True, exist_ok=True)
    (tmp_path / "3" / "model.eqx").write_bytes(b"")

    loader = ExperimentLoader(reader=_StubReader(cfg))
    steps = loader.list_checkpoints()
    assert steps == [0, 3]
    assert loader.latest_checkpoint() == 3

