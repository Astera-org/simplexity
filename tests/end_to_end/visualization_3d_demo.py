"""Hydra-powered demo that renders a 3D scatter plot via PlotConfig YAML."""

from __future__ import annotations

import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from simplexity.visualization.altair_renderer import build_altair_chart
from simplexity.visualization.data_registry import DictDataRegistry
from simplexity.visualization.plotly_renderer import build_plotly_figure
from simplexity.visualization.structured_configs import PlotConfig


@dataclass
class SyntheticDataConfig:
    """Configuration for generating synthetic 3D clusters."""

    source_name: str = "cloud"
    num_points: int = 600
    clusters: int = 4
    cluster_spread: float = 0.8
    seed: int = 11


@dataclass
class Scatter3DDemoConfig:
    """Root Hydra config for the demo."""

    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    output_html: str = "scatter3d_demo.html"


@hydra.main(version_base=None, config_path="configs/visualization", config_name="3d_scatter")
def main(cfg: DictConfig) -> None:
    """Main entry point for the demo."""
    data_cfg = _convert_cfg(cfg.data, SyntheticDataConfig)
    plot_cfg = _convert_cfg(cfg.plot, PlotConfig)
    output_html = cast(str, cfg.get("output_html", "scatter3d_demo.html"))
    dataframe = _generate_dataset(data_cfg)
    registry = DictDataRegistry({data_cfg.source_name: dataframe})

    if plot_cfg.backend == "plotly":
        figure = build_plotly_figure(plot_cfg, registry)
        _save_plotly_figure(figure, output_html)
    else:
        chart = build_altair_chart(plot_cfg, registry)
        _save_altair_chart(chart, output_html)

    print(f"Saved interactive plot to {output_html}")  # noqa: T201 - demo script output


def _generate_dataset(cfg: SyntheticDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    points_per_cluster = max(1, cfg.num_points // cfg.clusters)
    remainder = cfg.num_points % cfg.clusters
    records: list[dict[str, float | int | str]] = []
    for cluster_idx in range(cfg.clusters):
        center = rng.normal(0.0, cfg.cluster_spread * 3.0, size=3)
        count = points_per_cluster + (1 if cluster_idx < remainder else 0)
        for _ in range(count):
            noise = rng.normal(0.0, cfg.cluster_spread, size=3)
            x, y, z = center + noise
            magnitude = float(np.sqrt(x**2 + y**2 + z**2))
            records.append(
                {
                    "cluster": f"C{cluster_idx + 1}",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "magnitude": magnitude,
                }
            )
    return pd.DataFrame.from_records(records)


def _convert_cfg[T](cfg_section: DictConfig, schema: type[T]) -> T:
    """Convert DictConfig to dataclass instance, handling nested dataclasses recursively."""
    # Convert DictConfig to plain dict to avoid OmegaConf's Union/Literal type validation issues
    cfg_dict = OmegaConf.to_container(cfg_section, resolve=True) or {}
    return _dict_to_dataclass(cfg_dict, schema)


def _convert_value_by_type(value: Any, field_type: Any) -> Any:
    """Convert a value based on its expected type (handles lists, dataclasses, etc.)."""
    origin = get_origin(field_type)

    # Handle list types
    if origin is list:
        args = get_args(field_type)
        if isinstance(value, list) and args:
            item_type = args[0]
            if is_dataclass(item_type):
                return [
                    _dict_to_dataclass(item, item_type) if isinstance(item, dict) else item  # type: ignore[arg-type]
                    for item in value
                ]
        return value
    # Handle dataclass types
    if isinstance(value, dict) and is_dataclass(field_type):
        return _dict_to_dataclass(value, field_type)  # type: ignore[arg-type]

    return value


def _dict_to_dataclass(data: dict[str, Any] | Any, schema: type[Any]) -> Any:  # pylint: disable=too-many-branches
    """Recursively convert dict to dataclass instance, handling nested structures."""
    if not isinstance(data, dict):
        return data

    if not is_dataclass(schema):
        return data

    # Get field types from the dataclass schema, resolving string annotations
    try:
        field_types = get_type_hints(schema)
    except (TypeError, NameError):
        # Fallback to field.type if get_type_hints fails (e.g., forward references)
        field_types = {f.name: f.type for f in fields(schema)}

    # Convert nested dicts to their corresponding dataclass types
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_types:
            converted[key] = value
            continue

        field_type = field_types[key]
        origin = get_origin(field_type)

        # Handle Optional types (Union[X, None] or X | None)
        if origin is Union or origin is types.UnionType:
            args = get_args(field_type)
            # Handle Optional[X] -> Union[X, None]
            if args and len(args) == 2 and types.NoneType in args:
                if value is None:
                    converted[key] = None
                else:
                    non_none_type = next((t for t in args if t is not types.NoneType), None)
                    if non_none_type:
                        # Recursively handle the non-None type (could be a list, dict, etc.)
                        converted[key] = _convert_value_by_type(value, non_none_type)
                    else:
                        converted[key] = value
            elif args and isinstance(value, dict):
                # For other Union types, try to find a dataclass type that matches
                dataclass_type = next((t for t in args if is_dataclass(t)), None)
                if dataclass_type:
                    converted[key] = _dict_to_dataclass(value, dataclass_type)  # type: ignore[arg-type]
                else:
                    converted[key] = value
            else:
                # For other Union types, try to convert based on the first non-None type
                non_none_types = [t for t in args if t is not types.NoneType] if args else []
                if non_none_types and value is not None:
                    converted[key] = _convert_value_by_type(value, non_none_types[0])
                else:
                    converted[key] = value
        # Handle list types
        elif origin is list:
            args = get_args(field_type)
            if isinstance(value, list) and args:
                item_type = args[0]
                if is_dataclass(item_type):
                    converted[key] = [
                        _dict_to_dataclass(item, item_type) if isinstance(item, dict) else item  # type: ignore[arg-type]
                        for item in value
                    ]
                else:
                    converted[key] = value
            else:
                converted[key] = value
        # Handle direct dataclass types
        elif isinstance(value, dict) and is_dataclass(field_type):
            converted[key] = _dict_to_dataclass(value, field_type)  # type: ignore[arg-type]
        else:
            converted[key] = value

    return schema(**converted)


def _save_plotly_figure(figure, filename: str) -> None:
    output_path = Path(get_original_cwd()) / filename
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def _save_altair_chart(chart, filename: str) -> None:
    output_path = Path(get_original_cwd()) / filename
    chart.save(str(output_path))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
