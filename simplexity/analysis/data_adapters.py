"""Data adapters for converting analysis results to visualization-ready DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pca_results_to_dataframe(
    pca_results_by_step_and_layer: dict[int, dict[str, dict[str, Any]]],
    n_components: int = 2,
    include_beliefs: bool = True,
) -> pd.DataFrame:
    """Convert nested PCA results to a flat DataFrame for 2D/3D visualization.

    Args:
        pca_results_by_step_and_layer: Nested dict {step: {layer_name: pca_result}}
        n_components: Number of principal components to include (2 for 2D, 3 for 3D)
        include_beliefs: Whether to include belief columns (belief_0, belief_1, belief_2)

    Returns:
        DataFrame with columns: step, layer, point_id, pc1, pc2, [pc3], [belief_0, belief_1, belief_2],
            [sequence, next_token, seq_length] (if prefixes available)
    """
    rows = []
    for step, layer_dict in pca_results_by_step_and_layer.items():
        for layer_name, pca_res in layer_dict.items():
            X_proj = pca_res["X_proj"][:, :n_components]
            beliefs = pca_res.get("beliefs") if include_beliefs else None
            prefixes = pca_res.get("prefixes")

            for i in range(len(X_proj)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                }
                for comp_idx in range(n_components):
                    row[f"pc{comp_idx + 1}"] = X_proj[i, comp_idx]

                if beliefs is not None and beliefs.shape[1] == 3:
                    row["belief_0"] = beliefs[i, 0]
                    row["belief_1"] = beliefs[i, 1]
                    row["belief_2"] = beliefs[i, 2]

                # Add sequence metadata if available
                if prefixes is not None:
                    prefix = prefixes[i]
                    row["sequence"] = " ".join(str(t) for t in prefix)
                    row["next_token"] = prefix[-1] if prefix else None
                    row["seq_length"] = len(prefix)

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
        include_beliefs: Whether to include belief columns (belief_0, belief_1, belief_2)

    Returns:
        DataFrame with columns: step, layer, point_id, x, y, belief_type, [belief_0, belief_1, belief_2],
            [sequence, next_token, seq_length] (if prefixes available)
            where belief_type is either 'true' or 'predicted'
    """
    from simplexity.analysis.regression import project_to_simplex

    rows = []
    for step, layer_dict in regression_results_by_step_and_layer.items():
        for layer_name, reg_result in layer_dict.items():
            # Project true beliefs to simplex
            true_x, true_y = project_to_simplex(reg_result.true_values)
            true_beliefs = np.asarray(reg_result.true_values)

            for i in range(len(true_x)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": true_x[i],
                    "y": true_y[i],
                    "belief_type": "true",
                }

                if include_beliefs and true_beliefs.shape[1] == 3:
                    row["belief_0"] = true_beliefs[i, 0]
                    row["belief_1"] = true_beliefs[i, 1]
                    row["belief_2"] = true_beliefs[i, 2]

                # Add sequence metadata if available
                if reg_result.prefixes is not None:
                    prefix = reg_result.prefixes[i]
                    row["sequence"] = " ".join(str(t) for t in prefix)
                    row["next_token"] = prefix[-1] if prefix else None
                    row["seq_length"] = len(prefix)

                rows.append(row)

            # Project predicted beliefs to simplex
            pred_x, pred_y = project_to_simplex(reg_result.predictions)
            pred_beliefs = np.asarray(reg_result.predictions)

            for i in range(len(pred_x)):
                row = {
                    "step": step,
                    "layer": layer_name,
                    "point_id": i,
                    "x": pred_x[i],
                    "y": pred_y[i],
                    "belief_type": "predicted",
                }

                if include_beliefs and pred_beliefs.shape[1] == 3:
                    row["belief_0"] = pred_beliefs[i, 0]
                    row["belief_1"] = pred_beliefs[i, 1]
                    row["belief_2"] = pred_beliefs[i, 2]

                # Add sequence metadata if available
                if reg_result.prefixes is not None:
                    prefix = reg_result.prefixes[i]
                    row["sequence"] = " ".join(str(t) for t in prefix)
                    row["next_token"] = prefix[-1] if prefix else None
                    row["seq_length"] = len(prefix)

                rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "pca_results_to_dataframe",
    "variance_to_dataframe",
    "regression_results_to_dataframe",
]
