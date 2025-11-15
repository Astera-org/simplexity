"""Data adapters for converting analysis results to visualization-ready DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pca_results_to_dataframe(
    pca_results_by_step_and_layer: dict[int, dict[str, dict[str, Any]]],
    n_components: int = 2,
) -> pd.DataFrame:
    """Convert nested PCA results to a flat DataFrame for 2D/3D visualization.

    Args:
        pca_results_by_step_and_layer: Nested dict {step: {layer_name: pca_result}}
        n_components: Number of principal components to include (2 for 2D, 3 for 3D)

    Returns:
        DataFrame with columns: step, layer, point_id, pc1, pc2, [pc3]
    """
    rows = []
    for step, layer_dict in pca_results_by_step_and_layer.items():
        for layer_name, pca_res in layer_dict.items():
            X_proj = pca_res["X_proj"][:, :n_components]
            for i in range(len(X_proj)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                }
                for comp_idx in range(n_components):
                    row[f"pc{comp_idx + 1}"] = X_proj[i, comp_idx]
                rows.append(row)
    return pd.DataFrame(rows)


def variance_to_dataframe(
    pca_results_by_step_and_layer: dict[int, dict[str, dict[str, Any]]],
    max_components: int | None = None,
) -> pd.DataFrame:
    """Convert variance data to DataFrame for line plots.

    Args:
        pca_results_by_step_and_layer: Nested dict {step: {layer_name: pca_result}}
        max_components: Maximum number of components to include

    Returns:
        DataFrame with columns: step, layer, component, cumulative_variance
    """
    rows = []
    for step, layer_dict in pca_results_by_step_and_layer.items():
        for layer_name, pca_res in layer_dict.items():
            var_ratio = pca_res["explained_variance_ratio"]
            cumsum = np.cumsum(var_ratio)

            if max_components is not None:
                cumsum = cumsum[:max_components]

            for comp_idx, cum_var in enumerate(cumsum):
                rows.append({
                    "step": step,
                    "layer": layer_name,
                    "component": comp_idx + 1,
                    "cumulative_variance": cum_var,
                })
    return pd.DataFrame(rows)


def regression_results_to_dataframe(
    regression_results_by_step_and_layer: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Convert regression results to DataFrame for simplex projection visualization.

    Args:
        regression_results_by_step_and_layer: Nested dict {step: {layer_name: RegressionResult}}

    Returns:
        DataFrame with columns: step, layer, point_id, x, y, belief_type
            where belief_type is either 'true' or 'predicted'
    """
    rows = []
    for step, layer_dict in regression_results_by_step_and_layer.items():
        for layer_name, reg_result in layer_dict.items():
            # True beliefs
            true_proj = reg_result.projected_true_beliefs
            for i in range(len(true_proj)):
                rows.append({
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": true_proj[i, 0],
                    "y": true_proj[i, 1],
                    "belief_type": "true",
                })

            # Predicted beliefs
            pred_proj = reg_result.projected_predicted_beliefs
            for i in range(len(pred_proj)):
                rows.append({
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": pred_proj[i, 0],
                    "y": pred_proj[i, 1],
                    "belief_type": "predicted",
                })

    return pd.DataFrame(rows)


__all__ = [
    "pca_results_to_dataframe",
    "variance_to_dataframe",
    "regression_results_to_dataframe",
]
