"""Helpers for resolving logical visualization data sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd


class DataRegistry(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for registry objects that return pandas DataFrames."""

    def get(self, source_name: str) -> pd.DataFrame:
        """Return the DataFrame associated with ``source_name``."""
        ...  # pylint: disable=unnecessary-ellipsis


class DictDataRegistry:  # pylint: disable=too-few-public-methods
    """Simple registry backed by an in-memory mapping."""

    def __init__(self, data: Mapping[str, pd.DataFrame] | None = None) -> None:
        self._data: dict[str, pd.DataFrame] = dict(data or {})

    def get(self, source_name: str) -> pd.DataFrame:
        """Get the DataFrame associated with ``source_name``."""
        try:
            return self._data[source_name]
        except KeyError as exc:  # pragma: no cover - simple error wrapper
            raise ValueError(f"Data source '{source_name}' is not registered") from exc


def resolve_data_source(source_name: str, data_registry: DataRegistry | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve a logical source name regardless of the registry implementation."""
    if isinstance(data_registry, Mapping):
        if source_name not in data_registry:
            raise ValueError(f"Data source '{source_name}' is not registered")
        return data_registry[source_name]
    return data_registry.get(source_name)
