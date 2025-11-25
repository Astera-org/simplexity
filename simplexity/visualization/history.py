"""Utilities for persisting visualization history for interactive controls."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from simplexity.visualization.structured_configs import PlotConfig

LOGGER = logging.getLogger(__name__)

HISTORY_VERSION = 1
HISTORY_DIRNAME = "history"
HISTORY_DATA_SUFFIX = ".jsonl"
HISTORY_META_SUFFIX = ".meta.json"


def history_paths(root: Path, safe_name: str) -> tuple[Path, Path]:
    """Return the data and metadata file paths for a visualization history entry."""

    history_dir = root / HISTORY_DIRNAME
    data_path = history_dir / f"{safe_name}{HISTORY_DATA_SUFFIX}"
    meta_path = history_dir / f"{safe_name}{HISTORY_META_SUFFIX}"
    return data_path, meta_path


def plot_config_signature(plot_cfg: PlotConfig) -> str:
    """Create a stable hash of a PlotConfig to detect incompatible history files."""

    serialized = json.dumps(
        dataclasses.asdict(plot_cfg),
        sort_keys=True,
        default=_serialize_unknown,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_history_dataframe(data_path: Path, meta_path: Path, *, expected_signature: str) -> pd.DataFrame:
    """Load previously saved visualization dataframe if metadata matches signature."""

    if not data_path.exists() or not meta_path.exists():
        return pd.DataFrame()

    try:
        with meta_path.open(encoding="utf-8") as source:
            metadata = json.load(source)
    except json.JSONDecodeError:
        LOGGER.warning("Visualization history metadata at %s is corrupted; ignoring existing history.", meta_path)
        return pd.DataFrame()

    if metadata.get("version") != HISTORY_VERSION or metadata.get("signature") != expected_signature:
        LOGGER.info("Visualization history metadata at %s is outdated or mismatched; starting fresh.", meta_path)
        return pd.DataFrame()

    try:
        return pd.read_json(data_path, orient="records", lines=True)
    except ValueError:
        LOGGER.warning("Visualization history data at %s is corrupted; ignoring existing history.", data_path)
        return pd.DataFrame()


def save_history_dataframe(
    dataframe: pd.DataFrame,
    data_path: Path,
    meta_path: Path,
    *,
    signature: str,
    analysis: str,
    name: str,
    backend: str,
) -> None:
    """Persist visualization dataframe and metadata for future accumulation."""

    data_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_json(data_path, orient="records", lines=True)
    metadata = {
        "version": HISTORY_VERSION,
        "analysis": analysis,
        "name": name,
        "backend": backend,
        "signature": signature,
        "rows": len(dataframe),
    }
    with meta_path.open("w", encoding="utf-8") as sink:
        json.dump(metadata, sink, indent=2)


def _serialize_unknown(value: Any) -> str:
    """Best-effort serialization hook for dataclasses.asdict JSON dumps."""

    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = [
    "HISTORY_DIRNAME",
    "HISTORY_DATA_SUFFIX",
    "HISTORY_META_SUFFIX",
    "history_paths",
    "load_history_dataframe",
    "plot_config_signature",
    "save_history_dataframe",
]
