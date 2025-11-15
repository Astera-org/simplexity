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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get value from dict-like object (supports both dict and OmegaConf DictConfig)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif isinstance(obj, DictConfig):
        return obj.get(key, default)
    else:
        return getattr(obj, key, default)


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
    # Convert registry to dict if needed
    if isinstance(data_registry, dict):
        registry_dict = data_registry
    else:
        # It's a DataRegistry - convert to dict
        registry_dict = {name: data_registry.get(name) for name in dir(data_registry)}

    # Check if expansion is needed
    if _get(config_dict, "expands") is None:
        # No expansion needed, convert directly to PlotConfig
        return _dict_to_plot_config(config_dict)

    expand_configs = _get(config_dict, "expands")

    # Get the data source from config
    data_section = _get(config_dict, "data", {})
    if _get(data_section, "source") is None:
        raise ValueError("No data source specified in config")

    data_source = _get(data_section, "source")

    # Generate layers from all expand configs
    layers = []
    for expand_cfg in expand_configs:
        expand_by = _get(expand_cfg, "by")
        layer_name_pattern = _get(expand_cfg, "layer_name_pattern")
        template = _get(expand_cfg, "template")

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
            template_filters = _get(template, "filters")
            if template_filters:
                filters.extend(template_filters)

            # Create layer config
            layer = LayerConfig(
                name=layer_name,
                data=DataConfig(
                    source=data_source,
                    filters=filters,
                ),
                geometry=_dict_to_geometry_config(_get(template, "geometry")),
                aesthetics=_dict_to_aesthetics_config(_get(template, "aesthetics")),
            )
            layers.append(layer)

    # Build the full plot config
    # Use the top-level data source as default
    default_source = _get(data_section, "source", "main")

    plot_config = PlotConfig(
        backend=_get(config_dict, "backend", "plotly"),
        data=DataConfig(source=default_source),
        layers=layers,
        size=_dict_to_size_config(_get(config_dict, "size", {})),
        guides=_dict_to_guides_config(_get(config_dict, "guides", {})),
        background=_get(config_dict, "background"),
    )

    return plot_config


def _dict_to_plot_config(config_dict: dict[str, Any]) -> PlotConfig:
    """Convert a dict to PlotConfig (no expansion)."""
    from simplexity.visualization.structured_configs import PlotConfig

    # This is a simplified version - you may want more robust conversion
    return PlotConfig(**config_dict)


def _dict_to_geometry_config(geom_dict: dict[str, Any] | DictConfig):
    """Convert dict/DictConfig to GeometryConfig."""
    from simplexity.visualization.structured_configs import GeometryConfig

    return GeometryConfig(
        type=_get(geom_dict, "type"),
        props=_get(geom_dict, "props", {}),
    )


def _dict_to_aesthetics_config(aes_dict: dict[str, Any] | DictConfig):
    """Convert dict/DictConfig to AestheticsConfig."""
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
        scale_dict = _get(ch_dict, "scale")
        scale = None
        if scale_dict is not None:
            scale = ScaleConfig(
                type=_get(scale_dict, "type"),
                domain=_get(scale_dict, "domain"),
                range=_get(scale_dict, "range"),
                clamp=_get(scale_dict, "clamp"),
                nice=_get(scale_dict, "nice"),
                reverse=_get(scale_dict, "reverse"),
            )

        return ChannelAestheticsConfig(
            field=_get(ch_dict, "field"),
            type=_get(ch_dict, "type"),
            value=_get(ch_dict, "value"),
            aggregate=_get(ch_dict, "aggregate"),
            bin=_get(ch_dict, "bin"),
            time_unit=_get(ch_dict, "time_unit"),
            scale=scale,
            axis=_get(ch_dict, "axis"),
            legend=_get(ch_dict, "legend"),
            sort=_get(ch_dict, "sort"),
            title=_get(ch_dict, "title"),
        )

    # Build tooltip list if present
    tooltip = None
    tooltip_data = _get(aes_dict, "tooltip")
    if tooltip_data:
        tooltip_list = [to_channel(tt) for tt in tooltip_data]
        # Filter out None values
        tooltip = [t for t in tooltip_list if t is not None]
        if not tooltip:
            tooltip = None

    return AestheticsConfig(
        x=to_channel(_get(aes_dict, "x")) if _get(aes_dict, "x") else None,
        y=to_channel(_get(aes_dict, "y")) if _get(aes_dict, "y") else None,
        z=to_channel(_get(aes_dict, "z")) if _get(aes_dict, "z") else None,
        color=to_channel(_get(aes_dict, "color")) if _get(aes_dict, "color") else None,
        size=to_channel(_get(aes_dict, "size")) if _get(aes_dict, "size") else None,
        shape=to_channel(_get(aes_dict, "shape")) if _get(aes_dict, "shape") else None,
        opacity=to_channel(_get(aes_dict, "opacity")) if _get(aes_dict, "opacity") else None,
        tooltip=tooltip,
    )


def _dict_to_size_config(size_dict: dict[str, Any] | DictConfig):
    """Convert dict/DictConfig to PlotSizeConfig."""
    from simplexity.visualization.structured_configs import PlotSizeConfig

    if not size_dict:
        return PlotSizeConfig()

    return PlotSizeConfig(
        width=_get(size_dict, "width"),
        height=_get(size_dict, "height"),
    )


def _dict_to_guides_config(guides_dict: dict[str, Any] | DictConfig):
    """Convert dict/DictConfig to PlotLevelGuideConfig."""
    from simplexity.visualization.structured_configs import PlotLevelGuideConfig

    if not guides_dict:
        return PlotLevelGuideConfig()

    return PlotLevelGuideConfig(
        title=_get(guides_dict, "title"),
        x_title=_get(guides_dict, "x_title"),
        y_title=_get(guides_dict, "y_title"),
        z_title=_get(guides_dict, "z_title"),
    )


__all__ = ["expand_plot_config"]
