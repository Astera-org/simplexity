"""Expand plot configuration templates into full PlotConfigs.

This module handles the expansion of layer templates based on data dimensions,
allowing dynamic generation of layers from compact YAML specifications.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from simplexity.visualization.data_registry import DataRegistry
from simplexity.visualization.structured_configs import (
    DataConfig,
    LayerConfig,
    PlotConfig,
)


def expand_plot_config(
    config_dict: dict[str, Any],
    data_registry: DataRegistry | dict[str, pd.DataFrame],
) -> PlotConfig:
    """Expand a plot configuration with layer templates into a full PlotConfig.

    This function takes a configuration dictionary that may contain an 'expands' directive
    and generates all necessary layers based on the data dimensions.

    Args:
        config_dict: Plot configuration dictionary (potentially from YAML/Hydra)
        data_registry: Registry of DataFrames for resolving data sources

    Returns:
        Full PlotConfig with all layers expanded

    Example:
        >>> config = {
        ...     "backend": "plotly",
        ...     "data": {"source": "pca_data"},
        ...     "expands": [{
        ...         "by": ["step", "layer"],
        ...         "layer_name_pattern": "step_{step}__layer_{layer}",
        ...         "template": {
        ...             "geometry": {"type": "point"},
        ...             "aesthetics": {
        ...                 "x": {"field": "pc1", "type": "quantitative"},
        ...                 "y": {"field": "pc2", "type": "quantitative"},
        ...                 "color": {"field": "point_id", "type": "quantitative"}
        ...             }
        ...         }
        ...     }]
        ... }
        >>> plot_config = expand_plot_config(config, data_registry)
    """
    # Convert registry to dict if needed
    if isinstance(data_registry, dict):
        registry_dict = data_registry
    else:
        # It's a DataRegistry - convert to dict
        registry_dict = {name: data_registry.get(name) for name in dir(data_registry)}

    # Check if expansion is needed
    if "expands" not in config_dict:
        # No expansion needed, convert directly to PlotConfig
        return _dict_to_plot_config(config_dict)

    expand_configs = config_dict["expands"]

    # Get the data source from config
    if "source" not in config_dict.get("data", {}):
        raise ValueError("No data source specified in config")

    data_source = config_dict["data"]["source"]

    # Generate layers from all expand configs
    layers = []
    for expand_cfg in expand_configs:
        expand_by = expand_cfg["by"]
        layer_name_pattern = expand_cfg["layer_name_pattern"]
        template = expand_cfg["template"]

        df = registry_dict[data_source]

        # Find unique combinations of expansion dimensions
        if not all(dim in df.columns for dim in expand_by):
            missing = [dim for dim in expand_by if dim not in df.columns]
            raise ValueError(f"Expansion dimensions {missing} not found in data columns: {list(df.columns)}")

        unique_combos_df = df[expand_by].drop_duplicates()
        unique_combos = unique_combos_df.to_dict(orient="records")  # type: ignore[call-overload]

        # Generate layers for each combination
        for combo in unique_combos:
            # Format layer name
            layer_name = layer_name_pattern.format(**combo)

            # Build filters for this combination
            filters = [f"{dim} == {combo[dim]!r}" if isinstance(combo[dim], str) else f"{dim} == {combo[dim]}"
                       for dim in expand_by]

            # Add any additional filters from template
            if "filters" in template:
                filters.extend(template["filters"])

            # Create layer config
            layer = LayerConfig(
                name=layer_name,
                data=DataConfig(
                    source=data_source,
                    filters=filters,
                ),
                geometry=_dict_to_geometry_config(template["geometry"]),
                aesthetics=_dict_to_aesthetics_config(template["aesthetics"]),
            )
            layers.append(layer)

    # Build the full plot config
    # Use the top-level data source as default (first expand may override)
    default_source = config_dict.get("data", {}).get("source", "main")

    plot_config = PlotConfig(
        backend=config_dict.get("backend", "plotly"),
        data=DataConfig(source=default_source),
        layers=layers,
        size=_dict_to_size_config(config_dict.get("size", {})),
        guides=_dict_to_guides_config(config_dict.get("guides", {})),
        background=config_dict.get("background"),
    )

    return plot_config


def _dict_to_plot_config(config_dict: dict[str, Any]) -> PlotConfig:
    """Convert a dict to PlotConfig (no expansion)."""
    from simplexity.visualization.structured_configs import PlotConfig

    # This is a simplified version - you may want more robust conversion
    return PlotConfig(**config_dict)


def _dict_to_geometry_config(geom_dict: dict[str, Any]):
    """Convert dict to GeometryConfig."""
    from simplexity.visualization.structured_configs import GeometryConfig

    return GeometryConfig(
        type=geom_dict["type"],
        props=geom_dict.get("props", {}),
    )


def _dict_to_aesthetics_config(aes_dict: dict[str, Any]):
    """Convert dict to AestheticsConfig."""
    from simplexity.visualization.structured_configs import (
        AestheticsConfig,
        ChannelAestheticsConfig,
    )

    # Helper to convert channel dicts
    def to_channel(ch_dict):
        if ch_dict is None:
            return None
        return ChannelAestheticsConfig(**ch_dict)

    # Build tooltip list if present
    tooltip = None
    if "tooltip" in aes_dict and aes_dict["tooltip"]:
        tooltip_list = [to_channel(tt) for tt in aes_dict["tooltip"]]
        # Filter out None values
        tooltip = [t for t in tooltip_list if t is not None]
        if not tooltip:
            tooltip = None

    return AestheticsConfig(
        x=to_channel(aes_dict.get("x")),
        y=to_channel(aes_dict.get("y")),
        z=to_channel(aes_dict.get("z")),
        color=to_channel(aes_dict.get("color")),
        size=to_channel(aes_dict.get("size")),
        shape=to_channel(aes_dict.get("shape")),
        opacity=to_channel(aes_dict.get("opacity")),
        tooltip=tooltip,
    )


def _dict_to_size_config(size_dict: dict[str, Any]):
    """Convert dict to PlotSizeConfig."""
    from simplexity.visualization.structured_configs import PlotSizeConfig

    return PlotSizeConfig(**size_dict)


def _dict_to_guides_config(guides_dict: dict[str, Any]):
    """Convert dict to PlotLevelGuideConfig."""
    from simplexity.visualization.structured_configs import PlotLevelGuideConfig

    return PlotLevelGuideConfig(**guides_dict)


__all__ = ["expand_plot_config"]
