from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import pandas as pd
from omegaconf import DictConfig


class RunReader(Protocol):
    """Protocol for reading experiment run data from a tracking backend."""

    def get_config(self) -> DictConfig:
        """Return the saved run config as a DictConfig."""
        ...

    def get_params(self) -> dict[str, str]:
        """Return run parameters as a simple dict of strings."""
        ...

    def get_tags(self) -> dict[str, str]:
        """Return run tags as a simple dict of strings."""
        ...

    def get_metrics(self, pattern: str | None = None) -> pd.DataFrame:
        """Return metrics as a tidy DataFrame with columns: metric, step, value, timestamp."""
        ...

    def list_artifacts(self, path: str | None = None) -> list[str]:
        """List artifact paths stored for this run (relative paths)."""
        ...

    def download_artifact(self, path: str, dst: str | Path | None = None) -> Path:
        """Download an artifact relative path to a destination directory and return the local file path."""
        ...

