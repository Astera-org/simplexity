"""Analysis tools for computational mechanics."""

from simplexity.analysis.tracker import AnalysisTracker
from simplexity.analysis.pca import (
    compute_pca,
    compute_variance_thresholds,
    plot_pca_2d,
    plot_pca_3d,
    plot_cumulative_explained_variance,
    generate_pca_plots,
    pca_prefix_activations,
    plot_pca_2d_with_step_slider,
    plot_pca_2d_with_layer_dropdown,
    plot_pca_2d_with_step_and_layer,
)
from simplexity.analysis.regression import (
    RegressionResult,
    regress_with_kfold_rcond_cv,
    regress_activations_to_beliefs,
    project_to_simplex,
    plot_simplex_projection_with_step_slider,
    plot_simplex_projection_with_layer_dropdown,
    plot_simplex_projection_with_step_and_layer,
)

__all__ = [
    # Tracker
    "AnalysisTracker",
    # PCA
    "compute_pca",
    "compute_variance_thresholds",
    "plot_pca_2d",
    "plot_pca_3d",
    "plot_cumulative_explained_variance",
    "generate_pca_plots",
    "pca_prefix_activations",
    "plot_pca_2d_with_step_slider",
    "plot_pca_2d_with_layer_dropdown",
    "plot_pca_2d_with_step_and_layer",
    # Regression
    "RegressionResult",
    "regress_with_kfold_rcond_cv",
    "regress_activations_to_beliefs",
    "project_to_simplex",
    "plot_simplex_projection_with_step_slider",
    "plot_simplex_projection_with_layer_dropdown",
    "plot_simplex_projection_with_step_and_layer",
]
