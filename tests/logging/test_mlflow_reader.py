from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from simplexity.logging.mlflow_reader import MLflowRunReader


class _Entry:
    def __init__(self, path: str, is_dir: bool) -> None:
        self.path = path
        self.is_dir = is_dir


def test_mlflow_reader_metrics_and_config(tmp_path: Path):
    """Basic test of metrics DataFrame and config fallback search."""
    with patch("mlflow.MlflowClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock run with metric keys
        run = SimpleNamespace(
            data=SimpleNamespace(metrics={"loss": 0.1, "validation/loss": 0.2}, params={}, tags={})
        )
        mock_client.get_run.return_value = run

        # Metric history (list of objects with .step/.value/.timestamp)
        class M:
            def __init__(self, step, value, ts):
                self.step = step
                self.value = value
                self.timestamp = ts

        mock_client.get_metric_history.side_effect = lambda run_id, key: [M(0, 1.0, 1), M(1, 0.9, 2)]

        # list_artifacts recursive: "" -> [nested/], "nested" -> [config.yaml]
        def list_artifacts(run_id, rel):
            return [_Entry("nested", True)] if rel == "" else [_Entry("config.yaml", False)]

        mock_client.list_artifacts.side_effect = list_artifacts

        # download_artifacts: fail for root config, succeed for nested/config.yaml
        def download_artifacts(run_id, path, dst):
            out_dir = Path(dst)
            out_dir.mkdir(parents=True, exist_ok=True)
            if path == "config.yaml":
                raise RuntimeError("not at root")
            # nested/config.yaml
            local = out_dir / "config.yaml"
            local.write_text(
                "predictive_model:\n  name: test\n  instance:\n    _target_: simplexity.predictive_models.gru_rnn.build_gru_rnn\n    vocab_size: 2\n\n# minimal persistence section to avoid None-handling paths\npersistence: null\n"
            )
            return str(local)

        mock_client.download_artifacts.side_effect = download_artifacts

        reader = MLflowRunReader(run_id="abc", tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))

        # Metrics
        df = reader.get_metrics()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"metric", "step", "value", "timestamp"}
        assert set(df["metric"].unique()) == {"loss", "validation/loss"}

        # Config
        cfg = reader.get_config()
        assert cfg.predictive_model.name == "test"

