"""Structured configuration objects for activation visualizations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from omegaconf import OmegaConf

from simplexity.exceptions import ConfigValidationError
from simplexity.utils.dataclass_conversion import convert_to_dataclass
from simplexity.visualization.structured_configs import (
    DataConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)

FieldSource = Literal[
    "projections", "scalars", "belief_states", "weights", "metadata", "scalar_pattern", "scalar_history"
]
ReducerType = Literal["argmax", "l2_norm"]


@dataclass
class ScalarSeriesMapping:
    """Describe how to unfold indexed scalar metrics into long-format (tidy) dataframe.

    This is used for plotting scalar values over an index dimension (e.g., cumulative
    variance vs. component count). For adding scalar values as columns to existing data,
    use wildcard mappings instead: `mappings: {rmse: {source: scalars, key: "layer_0_rmse"}}`.
    """

    key_template: str
    index_field: str
    value_field: str
    index_values: list[int] | None = None

    def __post_init__(self) -> None:
        if "{layer}" not in self.key_template:
            raise ConfigValidationError("scalar_series.key_template must include '{layer}' placeholder")
        if "{index}" not in self.key_template:
            raise ConfigValidationError("scalar_series.key_template must include '{index}' placeholder")
        if self.index_values is not None and not self.index_values:
            raise ConfigValidationError("scalar_series.index_values must not be empty")


@dataclass
class ActivationVisualizationFieldRef:
    """Map a DataFrame column to a specific activation artifact."""

    source: FieldSource
    key: str | None = None
    component: int | str | None = None
    reducer: ReducerType | None = None
    group_as: str | list[str] | None = None
    factor: int | str | None = None  # For selecting factor in factored belief states (3D arrays)
    _group_value: str | None = None  # Internal: populated during key/factor pattern expansion

    def __post_init__(self) -> None:
        if self.source == "projections" and not self.key:
            raise ConfigValidationError("Projection field references must specify the `key` to read from.")
        if self.source == "scalars" and not self.key:
            raise ConfigValidationError("Scalar field references must specify the `key` to read from.")
        if self.source == "scalar_pattern" and not self.key:
            raise ConfigValidationError("Scalar pattern field references must specify the `key` to read from.")
        if self.source == "scalar_history" and not self.key:
            raise ConfigValidationError("Scalar history field references must specify the `key` to read from.")
        if self.source == "metadata" and not self.key:
            raise ConfigValidationError("Metadata field references must specify the `key` to read from.")

        if isinstance(self.component, str):
            if self.component != "*" and not self._is_valid_range(self.component):
                raise ConfigValidationError(f"Component pattern '{self.component}' invalid. Use '*' or 'N...M'")
            if self.source not in ("projections", "belief_states"):
                raise ConfigValidationError(
                    f"Component patterns only supported for projections/belief_states, not '{self.source}'"
                )

        # Validate key patterns for projections
        if self.source == "projections" and self.key:
            has_key_pattern = "*" in self.key or self._is_valid_range(self.key)
            # Key patterns require group_as to name the resulting column(s)
            if has_key_pattern and self.group_as is None:
                raise ConfigValidationError(
                    f"Projection key pattern '{self.key}' requires `group_as` to name the expanded column(s)"
                )

        # Validate factor field (only for belief_states)
        if self.factor is not None:
            if self.source != "belief_states":
                raise ConfigValidationError(f"`factor` is only supported for belief_states, not '{self.source}'")
            if isinstance(self.factor, str):
                has_factor_pattern = self.factor == "*" or self._is_valid_range(self.factor)
                if has_factor_pattern and self.group_as is None:
                    raise ConfigValidationError(
                        f"Factor pattern '{self.factor}' requires `group_as` to name the expanded column(s)"
                    )

        # Validate group_as
        if self.group_as is not None and self.source not in ("projections", "belief_states"):
            raise ConfigValidationError(
                f"`group_as` is only supported for projections/belief_states, not '{self.source}'"
            )

    @staticmethod
    def _is_valid_range(component: str) -> bool:
        """Check if string matches 'N...M' range pattern."""
        if "..." not in component:
            return False
        parts = component.split("...")
        if len(parts) != 2:
            return False
        try:
            start, end = int(parts[0]), int(parts[1])
            return start < end
        except ValueError:
            return False


@dataclass
class SamplingConfig:
    """Configuration for sampling DataFrame rows to limit visualization size.

    When max_points is set, the DataFrame is sampled down to at most max_points
    rows per facet group (e.g., per layer, factor, or data_type combination).
    This ensures even distribution across subplots.
    """

    max_points: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_points is not None and self.max_points <= 0:
            raise ConfigValidationError("sampling.max_points must be a positive integer")


@dataclass
class CombinedMappingSection:
    """A labeled section of field mappings for combining multiple data sources.

    Used to combine projections and ground truth belief states into a single
    DataFrame with a label column for faceting (e.g., row faceting by data_type).
    """

    label: str
    mappings: dict[str, ActivationVisualizationFieldRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mappings:
            raise ConfigValidationError(f"Combined mapping section '{self.label}' must have at least one mapping.")


@dataclass
class ActivationVisualizationDataMapping:
    """Describe how to build the pandas DataFrame prior to rendering."""

    mappings: dict[str, ActivationVisualizationFieldRef] = field(default_factory=dict)
    scalar_series: ScalarSeriesMapping | None = None
    combined: list[CombinedMappingSection] | None = None  # For combining multiple data sources
    combine_as: str | None = None  # Column name for section labels (e.g., "data_type")
    sampling: SamplingConfig | None = None  # Optional sampling to limit visualization size

    def __post_init__(self) -> None:
        has_mappings = bool(self.mappings)
        has_scalar_series = self.scalar_series is not None
        has_combined = self.combined is not None and len(self.combined) > 0

        if not has_mappings and not has_scalar_series and not has_combined:
            raise ConfigValidationError(
                "Activation visualization data mapping must include at least one of: "
                "mappings, scalar_series, or combined sections."
            )

        if has_combined:
            if has_mappings:
                raise ConfigValidationError(
                    "Cannot use both 'mappings' and 'combined' in the same data_mapping. "
                    "Use 'combined' for multi-source visualizations."
                )
            if self.combine_as is None:
                raise ConfigValidationError(
                    "'combine_as' is required when using 'combined' sections to specify the label column name."
                )


@dataclass
class ActivationVisualizationPreprocessStep:
    """Preprocessing directives applied after the base DataFrame is built."""

    type: Literal["project_to_simplex", "combine_rgb"]
    input_fields: list[str]
    output_fields: list[str]

    def __post_init__(self) -> None:
        # Check if any input fields contain patterns (wildcards or ranges)
        has_pattern = any("*" in field or "..." in field for field in self.input_fields)

        if self.type == "project_to_simplex":
            # Skip input validation if patterns present (will be validated at runtime)
            if not has_pattern and len(self.input_fields) != 3:
                raise ConfigValidationError("project_to_simplex requires exactly three input_fields.")
            if len(self.output_fields) != 2:
                raise ConfigValidationError("project_to_simplex requires exactly two output_fields.")
        elif self.type == "combine_rgb":
            # Skip input validation if patterns present (will be validated at runtime)
            if not has_pattern and len(self.input_fields) < 3:
                raise ConfigValidationError("combine_rgb requires at least three input_fields.")
            if len(self.output_fields) != 1:
                raise ConfigValidationError("combine_rgb requires exactly one output_field.")


@dataclass
class ActivationVisualizationControlsConfig:
    """Optional control metadata to drive interactive front-ends."""

    slider: str | None = None
    dropdown: str | None = None
    toggle: str | None = None
    cumulative: bool = False
    accumulate_steps: bool = False

    def __post_init__(self) -> None:
        if self.accumulate_steps and self.slider == "step":
            raise ConfigValidationError(
                "controls.accumulate_steps cannot be used together with slider targeting 'step'."
            )


@dataclass
class ActivationVisualizationConfig:
    """Full specification for an analysis-attached visualization."""

    name: str
    data_mapping: ActivationVisualizationDataMapping
    backend: str | None = None
    plot: PlotConfig | None = None
    layer: LayerConfig | None = None
    size: PlotSizeConfig | None = None
    guides: PlotLevelGuideConfig | None = None
    preprocessing: list[ActivationVisualizationPreprocessStep] = field(default_factory=list)
    controls: ActivationVisualizationControlsConfig | None = None

    def resolve_plot_config(self, default_backend: str) -> PlotConfig:
        """Return a PlotConfig constructed from either `plot` or shorthand fields."""
        if self.plot is not None:
            plot_cfg = self.plot
        elif self.layer is not None:
            plot_cfg = PlotConfig(
                backend=self.backend or default_backend,
                layers=[self.layer],
                size=self.size or PlotSizeConfig(),
                guides=self.guides or PlotLevelGuideConfig(),
            )
        else:
            raise ConfigValidationError(
                f"Visualization '{self.name}' must specify either a PlotConfig (`plot`) or a single `layer`."
            )

        if plot_cfg.data is None:
            plot_cfg.data = DataConfig(source="main")
        else:
            plot_cfg.data.source = plot_cfg.data.source or "main"
        plot_cfg.backend = self.backend or plot_cfg.backend
        if self.size is not None:
            plot_cfg.size = self.size
        if self.guides is not None:
            plot_cfg.guides = self.guides
        if any(step.type == "combine_rgb" for step in self.preprocessing) and plot_cfg.backend != "plotly":
            raise ConfigValidationError("combine_rgb preprocessing requires backend='plotly'")
        return plot_cfg


def build_activation_visualization_config(raw_cfg: Mapping[str, Any]) -> ActivationVisualizationConfig:
    """Recursively convert dictionaries/OmegaConf nodes to visualization dataclasses."""
    if isinstance(raw_cfg, ActivationVisualizationConfig):
        return raw_cfg

    # Handle both plain dicts and OmegaConf configs
    if isinstance(raw_cfg, dict):
        container = raw_cfg
    else:
        container = OmegaConf.to_container(raw_cfg, resolve=False)
    if isinstance(container, dict):
        config_dict = cast(dict[str, Any], container)
    else:
        config_dict = {}
    data_mapping_cfg = config_dict.get("data_mapping")
    if data_mapping_cfg is None:
        raise ConfigValidationError("Visualization config must include a data_mapping block.")
    if "scalar_series" in data_mapping_cfg and data_mapping_cfg["scalar_series"] is not None:
        data_mapping_cfg["scalar_series"] = convert_to_dataclass(data_mapping_cfg["scalar_series"], ScalarSeriesMapping)
    if "mappings" in data_mapping_cfg and data_mapping_cfg["mappings"] is not None:
        data_mapping_cfg["mappings"] = {
            key: convert_to_dataclass(value, ActivationVisualizationFieldRef)
            for key, value in data_mapping_cfg["mappings"].items()
        }
    if "combined" in data_mapping_cfg and data_mapping_cfg["combined"] is not None:
        converted_sections = []
        for section in data_mapping_cfg["combined"]:
            section_mappings = section.get("mappings", {})
            converted_mappings = {
                key: convert_to_dataclass(value, ActivationVisualizationFieldRef)
                for key, value in section_mappings.items()
            }
            converted_sections.append(CombinedMappingSection(label=section["label"], mappings=converted_mappings))
        data_mapping_cfg["combined"] = converted_sections
    if "sampling" in data_mapping_cfg and data_mapping_cfg["sampling"] is not None:
        data_mapping_cfg["sampling"] = convert_to_dataclass(data_mapping_cfg["sampling"], SamplingConfig)
    config_dict["data_mapping"] = convert_to_dataclass(data_mapping_cfg, ActivationVisualizationDataMapping)
    if "preprocessing" in config_dict:
        config_dict["preprocessing"] = [
            convert_to_dataclass(step, ActivationVisualizationPreprocessStep) for step in config_dict["preprocessing"]
        ]
    if "controls" in config_dict and config_dict["controls"] is not None:
        config_dict["controls"] = convert_to_dataclass(config_dict["controls"], ActivationVisualizationControlsConfig)
    if "plot" in config_dict and config_dict["plot"] is not None:
        config_dict["plot"] = convert_to_dataclass(config_dict["plot"], PlotConfig)
    if "layer" in config_dict and config_dict["layer"] is not None:
        config_dict["layer"] = convert_to_dataclass(config_dict["layer"], LayerConfig)
    if "size" in config_dict and config_dict["size"] is not None:
        config_dict["size"] = convert_to_dataclass(config_dict["size"], PlotSizeConfig)
    if "guides" in config_dict and config_dict["guides"] is not None:
        config_dict["guides"] = convert_to_dataclass(config_dict["guides"], PlotLevelGuideConfig)

    return convert_to_dataclass(config_dict, ActivationVisualizationConfig)


__all__ = [
    "ActivationVisualizationConfig",
    "ActivationVisualizationControlsConfig",
    "ActivationVisualizationDataMapping",
    "ActivationVisualizationFieldRef",
    "ActivationVisualizationPreprocessStep",
    "CombinedMappingSection",
    "SamplingConfig",
    "ScalarSeriesMapping",
    "build_activation_visualization_config",
]
