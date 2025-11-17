"""Data adapters for converting analysis results to visualization-ready DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def beliefs_to_rgb(beliefs: np.ndarray) -> list[str]:
    """Convert 3D belief vectors to RGB color strings.

    Args:
        beliefs: Array of shape (N, 3) with belief probabilities

    Returns:
        List of RGB color strings in format "rgb(r,g,b)"
    """
    if beliefs.shape[1] != 3:
        raise ValueError(f"Expected beliefs with 3 dimensions, got {beliefs.shape[1]}")

    rgb = (beliefs * 255).astype(int)
    rgb = np.clip(rgb, 0, 255)

    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def pca_results_to_dataframe(
    pca_results_by_step_and_layer: dict[int, dict[str, dict[str, Any]]],
    n_components: int = 2,
    include_beliefs: bool = True,
) -> pd.DataFrame:
    """Convert nested PCA results to a flat DataFrame for 2D/3D visualization.

    Args:
        pca_results_by_step_and_layer: Nested dict {step: {layer_name: pca_result}}
        n_components: Number of principal components to include (2 for 2D, 3 for 3D)
        include_beliefs: Whether to include belief columns (belief_0, belief_1, belief_2, rgb_color)

    Returns:
        DataFrame with columns: step, layer, point_id, pc1, pc2, [pc3], [belief_0, belief_1, belief_2, rgb_color]
    """
    rows = []
    for step, layer_dict in pca_results_by_step_and_layer.items():
        for layer_name, pca_res in layer_dict.items():
            X_proj = pca_res["X_proj"][:, :n_components]
            beliefs = pca_res.get("beliefs") if include_beliefs else None

            if beliefs is not None and beliefs.shape[1] == 3:
                rgb_colors = beliefs_to_rgb(np.asarray(beliefs))
            else:
                rgb_colors = None

            for i in range(len(X_proj)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                }
                for comp_idx in range(n_components):
                    row[f"pc{comp_idx + 1}"] = X_proj[i, comp_idx]

                if beliefs is not None and rgb_colors is not None and beliefs.shape[1] == 3:
                    row["belief_0"] = beliefs[i, 0]
                    row["belief_1"] = beliefs[i, 1]
                    row["belief_2"] = beliefs[i, 2]
                    row["rgb_color"] = rgb_colors[i]

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
                rows.append(
                    {
                        "step": step,
                        "layer": layer_name,
                        "component": comp_idx + 1,
                        "cumulative_variance": cum_var,
                    }
                )
    return pd.DataFrame(rows)


def regression_results_to_dataframe(
    regression_results_by_step_and_layer: dict[int, dict[str, Any]],
    include_beliefs: bool = True,
) -> pd.DataFrame:
    """Convert regression results to DataFrame for simplex projection visualization.

    Args:
        regression_results_by_step_and_layer: Nested dict {step: {layer_name: RegressionResult}}
        include_beliefs: Whether to include belief columns (belief_0, belief_1, belief_2, rgb_color)

    Returns:
        DataFrame with columns: step, layer, point_id, x, y, belief_type, [belief_0, belief_1, belief_2, rgb_color]
            where belief_type is either 'true' or 'predicted'
    """
    from simplexity.analysis.regression import project_to_simplex

    rows = []
    for step, layer_dict in regression_results_by_step_and_layer.items():
        for layer_name, reg_result in layer_dict.items():
            # Project true beliefs to simplex
            true_x, true_y = project_to_simplex(reg_result.true_values)
            true_beliefs = np.asarray(reg_result.true_values)

            if include_beliefs and true_beliefs.shape[1] == 3:
                true_rgb_colors = beliefs_to_rgb(true_beliefs)
            else:
                true_rgb_colors = None

            for i in range(len(true_x)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": true_x[i],
                    "y": true_y[i],
                    "belief_type": "true",
                }

                if include_beliefs and true_beliefs.shape[1] == 3 and true_rgb_colors is not None:
                    row["belief_0"] = true_beliefs[i, 0]
                    row["belief_1"] = true_beliefs[i, 1]
                    row["belief_2"] = true_beliefs[i, 2]
                    row["rgb_color"] = true_rgb_colors[i]

                rows.append(row)

            # Project predicted beliefs to simplex
            pred_x, pred_y = project_to_simplex(reg_result.predictions)
            pred_beliefs = np.asarray(reg_result.predictions)

            if include_beliefs and pred_beliefs.shape[1] == 3:
                pred_rgb_colors = beliefs_to_rgb(pred_beliefs)
            else:
                pred_rgb_colors = None

            for i in range(len(pred_x)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": pred_x[i],
                    "y": pred_y[i],
                    "belief_type": "predicted",
                }

                if include_beliefs and pred_beliefs.shape[1] == 3 and pred_rgb_colors is not None:
                    row["belief_0"] = pred_beliefs[i, 0]
                    row["belief_1"] = pred_beliefs[i, 1]
                    row["belief_2"] = pred_beliefs[i, 2]
                    row["rgb_color"] = pred_rgb_colors[i]

                rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "pca_results_to_dataframe",
    "variance_to_dataframe",
    "regression_results_to_dataframe",
]
