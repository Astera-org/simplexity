"""Programmatic generation of plot configurations for analysis visualizations."""

from __future__ import annotations

from simplexity.analysis.plot_style_configs import (
    PCA2DStyleConfig,
    PCA3DStyleConfig,
    RegressionStyleConfig,
    VarianceStyleConfig,
)
from simplexity.visualization.structured_configs import (
    AestheticsConfig,
    AxisConfig,
    ChannelAestheticsConfig,
    DataConfig,
    GeometryConfig,
    LayerConfig,
    PlotConfig,
    PlotLevelGuideConfig,
    PlotSizeConfig,
)


def generate_pca_2d_config(
    steps: list[int],
    layers: list[str],
    style: PCA2DStyleConfig | None = None,
) -> PlotConfig:
    """Generate plot config for 2D PCA with step slider and layer dropdown.

    Creates one layer per (step, layer) combination with naming pattern:
    "step_{step}__layer_{layer}"

    Args:
        steps: List of training steps
        layers: List of layer names
        style: Optional styling configuration (can be loaded from YAML)

    Returns:
        PlotConfig ready for build_plotly_figure()
    """
    if style is None:
        style = PCA2DStyleConfig()

    plot_layers = []

    for step in steps:
        for layer_name in layers:
            marker_config = {}
            if style.marker_size is not None:
                marker_config["size"] = style.marker_size

            # Use belief-based RGB colors if enabled, otherwise default to point_id coloring
            if style.use_belief_colors:
                color_config = ChannelAestheticsConfig(field="rgb_color", type="nominal")
            else:
                color_config = ChannelAestheticsConfig(field="point_id", type="quantitative")

            plot_layers.append(
                LayerConfig(
                    name=f"step_{step}__layer_{layer_name}",
                    data=DataConfig(source="pca_data", filters=[f"step == {step}", f"layer == '{layer_name}'"]),
                    geometry=GeometryConfig(type="point", props=marker_config),
                    aesthetics=AestheticsConfig(
                        x=ChannelAestheticsConfig(field="pc1", type="quantitative", title=style.pc1_title),
                        y=ChannelAestheticsConfig(field="pc2", type="quantitative", title=style.pc2_title),
                        color=color_config,
                        opacity=ChannelAestheticsConfig(value=style.opacity),
                    ),
                )
            )

    return PlotConfig(
        backend="plotly",
        data=DataConfig(source="pca_data"),
        layers=plot_layers,
        size=PlotSizeConfig(width=style.width, height=style.height),
        guides=PlotLevelGuideConfig(title=style.title),
        background=style.background,
    )


def generate_pca_3d_config(
    steps: list[int],
    layers: list[str],
    style: PCA3DStyleConfig | None = None,
) -> PlotConfig:
    """Generate plot config for 3D PCA with step slider and layer dropdown.

    Args:
        steps: List of training steps
        layers: List of layer names
        style: Optional styling configuration (can be loaded from YAML)

    Returns:
        PlotConfig ready for build_plotly_figure()
    """
    if style is None:
        style = PCA3DStyleConfig()

    plot_layers = []

    for step in steps:
        for layer_name in layers:
            marker_config = {}
            if style.marker_size is not None:
                marker_config["size"] = style.marker_size

            # Use belief-based RGB colors if enabled, otherwise default to point_id coloring
            if style.use_belief_colors:
                color_config = ChannelAestheticsConfig(field="rgb_color", type="nominal")
            else:
                color_config = ChannelAestheticsConfig(field="point_id", type="quantitative")

            plot_layers.append(
                LayerConfig(
                    name=f"step_{step}__layer_{layer_name}",
                    data=DataConfig(source="pca_data", filters=[f"step == {step}", f"layer == '{layer_name}'"]),
                    geometry=GeometryConfig(type="point", props=marker_config),
                    aesthetics=AestheticsConfig(
                        x=ChannelAestheticsConfig(field="pc1", type="quantitative", title=style.pc1_title),
                        y=ChannelAestheticsConfig(field="pc2", type="quantitative", title=style.pc2_title),
                        z=ChannelAestheticsConfig(field="pc3", type="quantitative", title=style.pc3_title),
                        color=color_config,
                        opacity=ChannelAestheticsConfig(value=style.opacity),
                    ),
                )
            )

    return PlotConfig(
        backend="plotly",
        data=DataConfig(source="pca_data"),
        layers=plot_layers,
        size=PlotSizeConfig(width=style.width, height=style.height),
        guides=PlotLevelGuideConfig(title=style.title),
        background=style.background,
    )


def generate_cumulative_variance_config(
    steps: list[int],
    layers: list[str],
    style: VarianceStyleConfig | None = None,
    group_by_step: bool = True,
) -> PlotConfig:
    """Generate plot config for cumulative variance with dropdown control.

    Args:
        steps: List of training steps
        layers: List of layer names
        style: Optional styling configuration (can be loaded from YAML)
        group_by_step: If True, dropdown selects step (shows all layers at that step).
                      If False, dropdown selects layer (shows all steps for that layer).

    Returns:
        PlotConfig ready for build_plotly_figure()
    """
    if style is None:
        style = VarianceStyleConfig()

    plot_layers = []

    if group_by_step:
        # One layer per (step, layer) - dropdown for steps, shows all layers at selected step
        for step in steps:
            for layer_name in layers:
                plot_layers.append(
                    LayerConfig(
                        name=f"step_{step}__layer_{layer_name}",
                        data=DataConfig(
                            source="variance_data", filters=[f"step == {step}", f"layer == '{layer_name}'"]
                        ),
                        geometry=GeometryConfig(type="line", props={"width": style.line_width}),
                        aesthetics=AestheticsConfig(
                            x=ChannelAestheticsConfig(
                                field="component", type="quantitative", title=style.component_title
                            ),
                            y=ChannelAestheticsConfig(
                                field="cumulative_variance",
                                type="quantitative",
                                title=style.variance_title,
                                axis=AxisConfig(format=style.axis_format),
                            ),
                            color=ChannelAestheticsConfig(field="layer", type="nominal"),
                        ),
                    )
                )
    else:
        # One layer per (layer, step) - dropdown for layers, shows all steps for selected layer
        for layer_name in layers:
            for step in steps:
                plot_layers.append(
                    LayerConfig(
                        name=f"step_{step}__layer_{layer_name}",
                        data=DataConfig(
                            source="variance_data", filters=[f"step == {step}", f"layer == '{layer_name}'"]
                        ),
                        geometry=GeometryConfig(type="line", props={"width": style.line_width}),
                        aesthetics=AestheticsConfig(
                            x=ChannelAestheticsConfig(
                                field="component", type="quantitative", title=style.component_title
                            ),
                            y=ChannelAestheticsConfig(
                                field="cumulative_variance",
                                type="quantitative",
                                title=style.variance_title,
                                axis=AxisConfig(format=style.axis_format),
                            ),
                            color=ChannelAestheticsConfig(value=f"step_{step}"),
                        ),
                    )
                )

    return PlotConfig(
        backend="plotly",
        data=DataConfig(source="variance_data"),
        layers=plot_layers,
        size=PlotSizeConfig(width=style.width, height=style.height),
        guides=PlotLevelGuideConfig(title=style.title),
        background=style.background,
    )


def generate_regression_config(
    steps: list[int],
    layers: list[str],
    style: RegressionStyleConfig | None = None,
) -> PlotConfig:
    """Generate plot config for regression simplex projection with step slider and layer dropdown.

    Creates two layers per (step, layer) combination - one for true beliefs, one for predicted.

    Args:
        steps: List of training steps
        layers: List of layer names
        style: Optional styling configuration (can be loaded from YAML)

    Returns:
        PlotConfig ready for build_plotly_figure()
    """
    if style is None:
        style = RegressionStyleConfig()

    plot_layers = []

    for step in steps:
        for layer_name in layers:
            marker_config = {}
            if style.marker_size is not None:
                marker_config["size"] = style.marker_size

            # Use belief-based RGB colors if enabled, otherwise use fixed colors
            if style.use_belief_colors:
                true_color_config = ChannelAestheticsConfig(field="rgb_color", type="nominal")
                pred_color_config = ChannelAestheticsConfig(field="rgb_color", type="nominal")
            else:
                true_color_config = ChannelAestheticsConfig(value=style.true_color)
                pred_color_config = ChannelAestheticsConfig(value=style.predicted_color)

            # True beliefs
            plot_layers.append(
                LayerConfig(
                    name=f"step_{step}__layer_{layer_name}",
                    data=DataConfig(
                        source="regression_data",
                        filters=[f"step == {step}", f"layer == '{layer_name}'", "belief_type == 'true'"],
                    ),
                    geometry=GeometryConfig(type="point", props=marker_config),
                    aesthetics=AestheticsConfig(
                        x=ChannelAestheticsConfig(field="x", type="quantitative", title=style.x_title),
                        y=ChannelAestheticsConfig(field="y", type="quantitative", title=style.y_title),
                        color=true_color_config,
                        opacity=ChannelAestheticsConfig(value=style.opacity),
                    ),
                )
            )

            # Predicted beliefs
            plot_layers.append(
                LayerConfig(
                    name=f"step_{step}__layer_{layer_name}",
                    data=DataConfig(
                        source="regression_data",
                        filters=[f"step == {step}", f"layer == '{layer_name}'", "belief_type == 'predicted'"],
                    ),
                    geometry=GeometryConfig(type="point", props=marker_config),
                    aesthetics=AestheticsConfig(
                        x=ChannelAestheticsConfig(field="x", type="quantitative", title=style.x_title),
                        y=ChannelAestheticsConfig(field="y", type="quantitative", title=style.y_title),
                        color=pred_color_config,
                        opacity=ChannelAestheticsConfig(value=style.opacity),
                    ),
                )
            )

    return PlotConfig(
        backend="plotly",
        data=DataConfig(source="regression_data"),
        layers=plot_layers,
        size=PlotSizeConfig(width=style.width, height=style.height),
        guides=PlotLevelGuideConfig(title=style.title),
        background=style.background,
    )


__all__ = [
    "generate_pca_2d_config",
    "generate_pca_3d_config",
    "generate_cumulative_variance_config",
    "generate_regression_config",
]
