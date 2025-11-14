import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from sklearn.model_selection import KFold

from simplexity.utils.analysis_utils import build_prefix_dataset
from plotly import graph_objs as go


@dataclass
class RegressionResult:
    best_rcond: float
    dist: float              # weighted sum of Euclidean errors
    r2: float
    mse: np.ndarray          # per-dimension MSE (weighted)
    mae: np.ndarray
    rmse: np.ndarray
    predictions: np.ndarray  # (N, B)
    true_values: np.ndarray  # (N, B)
    weights: np.ndarray      # (N,)
    per_rcond_cv_error: Dict[float, float]


def _weighted_regression(
    X: jax.Array,  # (N, D)
    Y: jax.Array,  # (N, B)
    weights: jax.Array,  # (N,)
    rcond: float,
) -> jax.Array:
    """
    Core: solve weighted linear regression with a bias term using SVD-based pseudoinverse.
    Returns beta of shape (D+1, B).
    """
    # normalize weights once for numerical stability
    w = weights / weights.sum()
    sqrt_w = jnp.expand_dims(jnp.sqrt(w), axis=1)  # (N, 1)

    # add bias
    ones = jnp.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = jnp.concatenate([ones, X], axis=1)  # (N, D+1)

    Xw = Xb * sqrt_w  # (N, D+1)
    Yw = Y * sqrt_w  # (N, B)

    # SVD of Xw
    U, S, Vh = jnp.linalg.svd(Xw, full_matrices=False)  # Xw = U S Vh

    # rcond threshold
    tol = rcond * S.max()
    S_inv = jnp.where(S > tol, 1.0 / S, jnp.zeros_like(S))

    # pinv(Xw) = V S_inv U^T
    pinv_Xw = (Vh.T * S_inv) @ U.T  # (D+1, N)

    beta = pinv_Xw @ Yw  # (D+1, B)
    return beta


def _compute_metrics(
    X: jax.Array, Y: jax.Array, weights: jax.Array, beta: jax.Array
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute norm_dist, r2, mse, mae, rmse and predictions.
    """
    N, D = X.shape
    B = Y.shape[1]

    ones = jnp.ones((N, 1), dtype=X.dtype)
    Xb = jnp.concatenate([ones, X], axis=1)
    Y_pred = Xb @ beta  # (N, B)

    # use original weights (not normalized) for distances & MSE/MAE
    w = weights
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive")

    # norm_dist: weighted sum of Euclidean errors
    dists = jnp.sqrt(jnp.sum((Y_pred - Y) ** 2, axis=1))  # (N,)
    dist = (dists * w).sum().item()

    # weighted MSE/MAE per belief dimension
    w_ = jnp.expand_dims(w, axis=1)  # (N, 1)
    sq_err = (Y_pred - Y) ** 2
    abs_err = jnp.abs(Y_pred - Y)

    mse = (sq_err * w_).sum(axis=0) / total_w
    mae = (abs_err * w_).sum(axis=0) / total_w
    rmse = jnp.sqrt(mse)

    # weighted R^2 using normalized weights
    wn = w / total_w
    sqrt_wn = jnp.expand_dims(jnp.sqrt(wn), axis=1)
    Yw = Y * sqrt_wn
    Y_pred_w = Y_pred * sqrt_wn
    mean_Yw = (Yw).sum(axis=0) / sqrt_wn.sum()

    total_var = jnp.sum((Yw - mean_Yw) ** 2)
    expl_var = jnp.sum((Y_pred_w - mean_Yw) ** 2)
    r2 = (expl_var / total_var).item() if total_var > 0 else 0.0

    return (
        dist,
        r2,
        np.asarray(mse),
        np.asarray(mae),
        np.asarray(rmse),
        np.asarray(Y_pred),
    )


def regress_with_kfold_rcond_cv(
    X: jax.Array,  # (N, D)
    Y: jax.Array,  # (N, B)
    weights: jax.Array,  # (N,)
    rcond_values: Iterable[float],
    n_splits: int = 10,
    random_state: int = 42,
) -> RegressionResult:
    """K-fold CV over rcond, then fit final model on all data using best rcond."""
    N = X.shape[0]

    # convert splits to numpy for sklearn KFold
    indices = np.arange(N)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    per_rcond_cv_error: Dict[float, float] = {}
    best_rcond = None
    best_cv_error = float("inf")

    # --------- Cross-validation over rcond ---------
    for rcond in rcond_values:
        fold_errors = []

        for train_idx_np, test_idx_np in kf.split(indices):
            X_train, Y_train, w_train = X[train_idx_np], Y[train_idx_np], weights[train_idx_np]
            X_test, Y_test, w_test = X[test_idx_np], Y[test_idx_np], weights[test_idx_np]

            # fit on train
            beta = _weighted_regression(X_train, Y_train, w_train, rcond=rcond)

            # compute test error (norm_dist) on test
            ones_test = jnp.ones((X_test.shape[0], 1), dtype=X_test.dtype)
            Xb_test = jnp.concatenate([ones_test, X_test], axis=1)
            Y_pred_test = Xb_test @ beta

            dists = jnp.sqrt(jnp.sum((Y_pred_test - Y_test) ** 2, axis=1))
            fold_error = (dists * w_test).sum().item()
            fold_errors.append(fold_error)

        avg_error = float(np.mean(fold_errors))
        per_rcond_cv_error[rcond] = avg_error

        if avg_error < best_cv_error:
            best_cv_error = avg_error
            best_rcond = rcond

    if best_rcond is None:
        raise RuntimeError("No valid rcond found during CV")

    # --------- Final fit on all data ---------
    beta_final = _weighted_regression(X, Y, weights, rcond=best_rcond)

    dist, r2, mse, mae, rmse, Y_pred_np = _compute_metrics(X, Y, weights, beta_final)

    return RegressionResult(
        best_rcond=best_rcond,
        dist=dist,
        r2=r2,
        mse=mse,
        mae=mae,
        rmse=rmse,
        predictions=Y_pred_np,
        true_values=np.asarray(Y),
        weights=np.asarray(weights),
        per_rcond_cv_error=per_rcond_cv_error,
    )

def project_to_simplex(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y


def regress_activations_to_beliefs(
    inputs: jax.Array,  # (batch, seq_len)
    beliefs: jax.Array,  # (batch, seq_len, B)
    probs: jax.Array,  # (batch, seq_len)
    activations_by_layer: Dict[str, jax.Array],  # layer -> (batch, seq_len, d_layer)
    layer_name: str,
    rcond_values: Iterable[float] | None = None,
    n_splits: int = 10,
    random_state: int = 42,
    plot_projection: bool = False,
) -> Tuple[RegressionResult, go.Figure | None]:
    """
    Regress from activations at a given layer to beliefs using K-fold CV over rcond.
    """
    if rcond_values is None:
        rcond_values = [1e-15, 1e-10, 1e-5] + np.logspace(-8, -3, 50).tolist()

    prefix_dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)
    X = prefix_dataset.activations_by_layer[layer_name]  # (N, d_layer)
    Y = prefix_dataset.beliefs  # (N, B)
    weights = prefix_dataset.probs  # (N,)

    result = regress_with_kfold_rcond_cv(
        X,
        Y,
        weights,
        rcond_values,
        n_splits=n_splits,
        random_state=random_state,
    )

    projection_plot = None
    if plot_projection:
        # Project true and predicted beliefs to 2D simplex
        Y_pred_np = result.predictions
        Y_true_np = np.asarray(Y)

        x_pred, y_pred = project_to_simplex(Y_pred_np)
        x_true, y_true = project_to_simplex(Y_true_np)

        projection_plot = go.Figure()
        projection_plot.add_trace(
            go.Scatter(
                x=x_true,
                y=y_true,
                mode="markers",
                name="True Beliefs",
                marker=dict(color="blue", size=5, opacity=0.5),
            )
        )
        projection_plot.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="markers",
                name="Predicted Beliefs",
                marker=dict(color="red", size=5, opacity=0.5),
            )
        )
        projection_plot.update_layout(
            title=f"Belief Projections on 2-Simplex at Layer {layer_name}",
            xaxis_title="X",
            yaxis_title="Y",
            width=600,
            height=600,
        )

    return result, projection_plot


def plot_simplex_projection_with_step_slider(
    regression_results_by_step: Dict[int, RegressionResult],
    title: str = "Belief Projections on 2-Simplex Over Training Steps",
) -> go.Figure:
    """
    Create interactive simplex projection plot with slider to navigate through training steps.

    Args:
        regression_results_by_step: dict mapping step -> RegressionResult
        title: plot title

    Returns:
        Plotly Figure with step slider
    """
    steps_sorted = sorted(regression_results_by_step.keys())

    if not steps_sorted:
        raise ValueError("regression_results_by_step must not be empty")

    fig = go.Figure()

    # Create traces for each step (initially all hidden)
    for step in steps_sorted:
        result = regression_results_by_step[step]
        Y_pred_np = result.predictions
        Y_true_np = result.true_values

        x_pred, y_pred = project_to_simplex(Y_pred_np)
        x_true, y_true = project_to_simplex(Y_true_np)

        # Add true beliefs trace
        fig.add_trace(
            go.Scatter(
                x=x_true,
                y=y_true,
                mode="markers",
                name="True Beliefs",
                marker=dict(color="blue", size=5, opacity=0.5),
                visible=False,
            )
        )

        # Add predicted beliefs trace
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="markers",
                name="Predicted Beliefs",
                marker=dict(color="red", size=5, opacity=0.5),
                visible=False,
            )
        )

    # Make first step visible (both traces)
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create slider steps
    slider_steps = []
    for i, step in enumerate(steps_sorted):
        # Each step has 2 traces (true and predicted)
        visible_flags = []
        for j in range(len(steps_sorted)):
            visible_flags.append(j == i)  # true beliefs
            visible_flags.append(j == i)  # predicted beliefs

        step_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} (Step {step})"},
            ],
            "label": str(step),
        }
        slider_steps.append(step_dict)

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }
    ]

    fig.update_layout(
        sliders=sliders,
        title=f"{title} (Step {steps_sorted[0]})",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig


def plot_simplex_projection_with_layer_dropdown(
    regression_results_by_layer: Dict[str, RegressionResult],
    title: str = "Belief Projections on 2-Simplex Across Layers",
) -> go.Figure:
    """
    Create interactive simplex projection plot with dropdown to switch between layers.

    Args:
        regression_results_by_layer: dict mapping layer_name -> RegressionResult
        title: plot title

    Returns:
        Plotly Figure with layer dropdown
    """
    layers_sorted = sorted(regression_results_by_layer.keys())

    if not layers_sorted:
        raise ValueError("regression_results_by_layer must not be empty")

    fig = go.Figure()

    # Create traces for each layer (initially all hidden)
    for layer_name in layers_sorted:
        result = regression_results_by_layer[layer_name]
        Y_pred_np = result.predictions
        Y_true_np = result.true_values

        x_pred, y_pred = project_to_simplex(Y_pred_np)
        x_true, y_true = project_to_simplex(Y_true_np)

        # Add true beliefs trace
        fig.add_trace(
            go.Scatter(
                x=x_true,
                y=y_true,
                mode="markers",
                name="True Beliefs",
                marker=dict(color="blue", size=5, opacity=0.5),
                visible=False,
            )
        )

        # Add predicted beliefs trace
        fig.add_trace(
            go.Scatter(
                x=x_pred,
                y=y_pred,
                mode="markers",
                name="Predicted Beliefs",
                marker=dict(color="red", size=5, opacity=0.5),
                visible=False,
            )
        )

    # Make first layer visible (both traces)
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create dropdown buttons
    buttons = []
    for i, layer_name in enumerate(layers_sorted):
        # Each layer has 2 traces (true and predicted)
        visible_flags = []
        for j in range(len(layers_sorted)):
            visible_flags.append(j == i)  # true beliefs
            visible_flags.append(j == i)  # predicted beliefs

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} ({layer_name})"},
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} ({layers_sorted[0]})",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig


def plot_simplex_projection_with_step_and_layer(
    regression_results_by_step_and_layer: Dict[int, Dict[str, RegressionResult]],
    title: str = "Belief Projections on 2-Simplex",
) -> go.Figure:
    """
    Create interactive simplex projection plot with both step slider and layer dropdown.

    Args:
        regression_results_by_step_and_layer: dict mapping step -> layer_name -> RegressionResult
        title: plot title

    Returns:
        Plotly Figure with step slider and layer dropdown
    """
    steps_sorted = sorted(regression_results_by_step_and_layer.keys())
    if not steps_sorted:
        raise ValueError("regression_results_by_step_and_layer must not be empty")

    # Get all layer names (assume same layers at each step)
    layers_sorted = sorted(regression_results_by_step_and_layer[steps_sorted[0]].keys())

    fig = go.Figure()

    # Create traces for each (step, layer) combination
    for step in steps_sorted:
        for layer_name in layers_sorted:
            result = regression_results_by_step_and_layer[step][layer_name]
            Y_pred_np = result.predictions
            Y_true_np = result.true_values

            x_pred, y_pred = project_to_simplex(Y_pred_np)
            x_true, y_true = project_to_simplex(Y_true_np)

            # Add true beliefs trace
            fig.add_trace(
                go.Scatter(
                    x=x_true,
                    y=y_true,
                    mode="markers",
                    name="True Beliefs",
                    marker=dict(color="blue", size=5, opacity=0.5),
                    visible=False,
                )
            )

            # Add predicted beliefs trace
            fig.add_trace(
                go.Scatter(
                    x=x_pred,
                    y=y_pred,
                    mode="markers",
                    name="Predicted Beliefs",
                    marker=dict(color="red", size=5, opacity=0.5),
                    visible=False,
                )
            )

    # Make first trace visible (both traces for first step/layer)
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create slider for steps
    slider_steps = []
    for step_idx, step in enumerate(steps_sorted):
        visible_flags = []
        for s_idx in range(len(steps_sorted)):
            for l_idx in range(len(layers_sorted)):
                # Each (step, layer) has 2 traces
                # Show only traces for this step with the first layer
                visible_flags.append(s_idx == step_idx and l_idx == 0)
                visible_flags.append(s_idx == step_idx and l_idx == 0)

        step_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} (Step {step}, {layers_sorted[0]})"},
            ],
            "label": str(step),
        }
        slider_steps.append(step_dict)

    # Create dropdown for layers
    buttons = []
    for layer_idx, layer_name in enumerate(layers_sorted):
        visible_flags = []
        for s_idx in range(len(steps_sorted)):
            for l_idx in range(len(layers_sorted)):
                # Show only traces for the first step with this layer
                visible_flags.append(s_idx == 0 and l_idx == layer_idx)
                visible_flags.append(s_idx == 0 and l_idx == layer_idx)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {"title": f"{title} (Step {steps_sorted[0]}, {layer_name})"},
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "y": -0.1,
            "xanchor": "left",
            "currentvalue": {"prefix": "Step: ", "visible": True, "xanchor": "center"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "steps": slider_steps,
        }
    ]

    fig.update_layout(
        sliders=sliders,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        title=f"{title} (Step {steps_sorted[0]}, {layers_sorted[0]})",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig