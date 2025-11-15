"""Configuration for expanding plot templates into multiple layers.

This module supports the expansion of layer templates based on data dimensions,
allowing dynamic generation of layers from YAML configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simplexity.visualization.structured_configs import AestheticsConfig, GeometryConfig


@dataclass
class LayerExpansionConfig:
    """Configuration for expanding a single layer template into multiple layers.

    This allows YAML configs to specify a template that gets expanded for each
    unique combination of dimension values (e.g., each step/layer pair).
    """

    by: list[str]  # Dimension columns to expand by (e.g., ["step", "layer"])
    layer_name_pattern: str  # Pattern for layer names (e.g., "step_{step}__layer_{layer}")
    template: LayerTemplateConfig  # Template for each expanded layer


@dataclass
class LayerTemplateConfig:
    """Template for generating individual layers during expansion."""

    geometry: GeometryConfig
    aesthetics: AestheticsConfig
    filters: list[str] = field(default_factory=list)  # Additional filters beyond expand dimensions


__all__ = [
    "LayerExpansionConfig",
    "LayerTemplateConfig",
]
