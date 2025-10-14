from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

from omegaconf import OmegaConf


@contextmanager
def load_resolved_config(mlflow_client, run_id: str, artifact_path: str = "config.yaml") -> Iterator[Tuple[object, Path]]:
    """Download and load a resolved Hydra config from MLflow.

    Yields (resolved_cfg: DictConfig, local_path: Path) while a temporary
    directory exists. If `artifact_path` points to a directory, attempts to
    load `config.yaml` within it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(mlflow_client.download_artifacts(run_id, artifact_path, dst_path=tmpdir))
        if p.is_dir():
            p = p / "config.yaml"
        resolved = OmegaConf.load(str(p))
        yield resolved, p

