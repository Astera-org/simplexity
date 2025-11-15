"""Expand plot configuration templates into full PlotConfigs.

This module handles the expansion of layer templates based on data dimensions,
allowing dynamic generation of layers from compact YAML specifications.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from simplexity.visualization.data_registry import DataRegistry
from simplexity.visualization.structured_configs import (
    DataConfig,
    LayerConfig,
    PlotConfig,
)


def expand_plot_config(
    config_dict: dict[str, Any] | DictConfig,
    data_registry: DataRegistry | dict[str, pd.DataFrame],
) -> PlotConfig:
    """Expand a plot configuration with layer templates into a full PlotConfig.

    This function takes a configuration dictionary (or OmegaConf DictConfig) that may
    contain an 'expands' directive and generates all necessary layers based on the data dimensions.

    Args:
        config_dict: Plot configuration dict or OmegaConf DictConfig (from YAML/Hydra)
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
    from omegaconf import OmegaConf

    # Convert OmegaConf to plain dict if needed (dataclasses need plain dicts)
    if isinstance(config_dict, DictConfig):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    # Convert registry to dict if needed
    if isinstance(data_registry, dict):
        registry_dict = data_registry
    else:
        # It's a DataRegistry - convert to dict
        registry_dict = {name: data_registry.get(name) for name in dir(data_registry)}

    # Check if expansion is needed (config_dict is now a plain dict)
    if "expands" not in config_dict:
        # No expansion needed, convert directly to PlotConfig
        return _dict_to_plot_config(config_dict)

    expand_configs = config_dict["expands"]

    # Get the data source from config
    data_section = config_dict.get("data", {})
    if "source" not in data_section:
        raise ValueError("No data source specified in config")

    data_source = data_section["source"]

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
    # Use the top-level data source as default
    default_source = data_section.get("source", "main")

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
        ScaleConfig,
    )

    # Helper to convert channel dicts
    def to_channel(ch_dict):
        if ch_dict is None:
            return None
        # Handle scale if present
        scale_dict = ch_dict.get("scale")
        scale = None
        if scale_dict is not None:
            scale = ScaleConfig(
                type=scale_dict.get("type"),
                domain=scale_dict.get("domain"),
                range=scale_dict.get("range"),
                clamp=scale_dict.get("clamp"),
                nice=scale_dict.get("nice"),
                reverse=scale_dict.get("reverse"),
            )

        return ChannelAestheticsConfig(
            field=ch_dict.get("field"),
            type=ch_dict.get("type"),
            value=ch_dict.get("value"),
            aggregate=ch_dict.get("aggregate"),
            bin=ch_dict.get("bin"),
            time_unit=ch_dict.get("time_unit"),
            scale=scale,
            axis=ch_dict.get("axis"),
            legend=ch_dict.get("legend"),
            sort=ch_dict.get("sort"),
            title=ch_dict.get("title"),
        )

    # Build tooltip list if present
    tooltip = None
    tooltip_data = aes_dict.get("tooltip")
    if tooltip_data:
        tooltip_list = [to_channel(tt) for tt in tooltip_data]
        # Filter out None values
        tooltip = [t for t in tooltip_list if t is not None]
        if not tooltip:
            tooltip = None

    return AestheticsConfig(
        x=to_channel(aes_dict.get("x")) if aes_dict.get("x") else None,
        y=to_channel(aes_dict.get("y")) if aes_dict.get("y") else None,
        z=to_channel(aes_dict.get("z")) if aes_dict.get("z") else None,
        color=to_channel(aes_dict.get("color")) if aes_dict.get("color") else None,
        size=to_channel(aes_dict.get("size")) if aes_dict.get("size") else None,
        shape=to_channel(aes_dict.get("shape")) if aes_dict.get("shape") else None,
        opacity=to_channel(aes_dict.get("opacity")) if aes_dict.get("opacity") else None,
        tooltip=tooltip,
    )


def _dict_to_size_config(size_dict: dict[str, Any]):
    """Convert dict to PlotSizeConfig."""
    from simplexity.visualization.structured_configs import PlotSizeConfig

    if not size_dict:
        return PlotSizeConfig()

    return PlotSizeConfig(
        width=size_dict.get("width"),
        height=size_dict.get("height"),
    )


def _dict_to_guides_config(guides_dict: dict[str, Any]):
    """Convert dict to PlotLevelGuideConfig."""
    from simplexity.visualization.structured_configs import PlotLevelGuideConfig

    if not guides_dict:
        return PlotLevelGuideConfig()

    return PlotLevelGuideConfig(
        title=guides_dict.get("title"),
        subtitle=guides_dict.get("subtitle"),
        caption=guides_dict.get("caption"),
        labels=guides_dict.get("labels"),
    )


__all__ = ["expand_plot_config"]
