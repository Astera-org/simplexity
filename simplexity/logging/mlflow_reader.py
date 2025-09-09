from __future__ import annotations

import fnmatch
import tempfile
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd
from mlflow.entities import Metric
from omegaconf import DictConfig, OmegaConf

from simplexity.logging.run_reader import RunReader


class MLflowRunReader(RunReader):
    """Read experiment data for a single MLflow run."""

    def __init__(self, run_id: str, tracking_uri: str | None = None) -> None:
        self._run_id = run_id
        # The global mlflow.set_tracking_uri also works, but we keep it local
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        self._temp_dir = tempfile.TemporaryDirectory()

    # --- Basic run info helpers ---
    def _get_run(self):
        return self._client.get_run(self._run_id)

    # --- Config/params/tags ---
    def get_config(self) -> DictConfig:
        # We expect the config artifact to be saved as "config.yaml" at the root
        dst_dir = Path(self._temp_dir.name) / "config"
        dst_dir.mkdir(parents=True, exist_ok=True)
        local_path = self._client.download_artifacts(self._run_id, "config.yaml", str(dst_dir))
        return OmegaConf.load(local_path)

    def get_params(self) -> dict[str, str]:
        run = self._get_run()
        # mlflow returns a dict-like mapping of strings
        return dict(run.data.params)

    def get_tags(self) -> dict[str, str]:
        run = self._get_run()
        return dict(run.data.tags)

    # --- Metrics ---
    def get_metrics(self, pattern: str | None = None) -> pd.DataFrame:
        run = self._get_run()
        metric_keys = list(run.data.metrics.keys())
        if pattern:
            # Support glob-style filtering (e.g., "validation/*")
            metric_keys = [k for k in metric_keys if fnmatch.fnmatch(k, pattern)]

        records: list[tuple[str, int, float, int]] = []
        for key in metric_keys:
            history: list[Metric] = self._client.get_metric_history(self._run_id, key)
            for m in history:
                records.append((key, int(m.step), float(m.value), int(m.timestamp)))
        if not records:
            return pd.DataFrame(columns=["metric", "step", "value", "timestamp"])
        df = pd.DataFrame.from_records(records, columns=["metric", "step", "value", "timestamp"])
        df.sort_values(["metric", "step"], inplace=True)
        return df

    # --- Artifacts ---
    def list_artifacts(self, path: str | None = None) -> list[str]:
        """List artifact relative paths (recursively)."""
        base = path or ""
        results: list[str] = []

        def _recurse(rel: str) -> None:
            for e in self._client.list_artifacts(self._run_id, rel):
                child = f"{rel}/{e.path}" if rel else e.path
                if e.is_dir:
                    _recurse(child)
                else:
                    results.append(child)

        _recurse(base)
        return results

    def download_artifact(self, path: str, dst: str | Path | None = None) -> Path:
        dst_dir = Path(dst) if dst is not None else Path(self._temp_dir.name) / "artifacts"
        dst_dir.mkdir(parents=True, exist_ok=True)
        local_path = self._client.download_artifacts(self._run_id, path, str(dst_dir))
        return Path(local_path)

    # --- Cleanup ---
    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass

