from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from simplexity.utils.analysis_utils import build_prefix_dataset


@dataclass
class RegressionResult:
    """Results from regression analysis with optional sequence metadata."""

    best_rcond: float
    dist: float              # weighted sum of Euclidean errors
    r2: float
    mse: np.ndarray          # per-dimension MSE (weighted)
    mae: np.ndarray
    rmse: np.ndarray
    predictions: np.ndarray  # (N, B)
    true_values: np.ndarray  # (N, B)
    weights: np.ndarray      # (N,)
    per_rcond_cv_error: dict[float, float]
    prefixes: list[tuple[int, ...]] | None = None  # Sequence-so-far for each point


def _weighted_regression(
    X: jax.Array,  # (N, D)
    Y: jax.Array,  # (N, B)
    weights: jax.Array,  # (N,)
    rcond: float,
) -> jax.Array:
    """Solve weighted linear regression with a bias term using SVD-based pseudoinverse.

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
    S_inv = jnp.where(tol < S, 1.0 / S, jnp.zeros_like(S))

    # pinv(Xw) = V S_inv U^T
    pinv_Xw = (Vh.T * S_inv) @ U.T  # (D+1, N)

    beta = pinv_Xw @ Yw  # (D+1, B)
    return beta


def _compute_metrics(
    X: jax.Array, Y: jax.Array, weights: jax.Array, beta: jax.Array
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute norm_dist, r2, mse, mae, rmse and predictions."""
    N, D = X.shape

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
    prefixes: list[tuple[int, ...]] | None = None,
) -> RegressionResult:
    """K-fold CV over rcond, then fit final model on all data using best rcond."""
    N = X.shape[0]

    # convert splits to numpy for sklearn KFold
    indices = np.arange(N)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    per_rcond_cv_error: dict[float, float] = {}
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
        prefixes=prefixes,
    )


def regress_simple_sklearn(
    X: jax.Array,  # (N, D)
    Y: jax.Array,  # (N, B)
    weights: jax.Array,  # (N,)
    prefixes: list[tuple[int, ...]] | None = None,
) -> RegressionResult:
    """Simple weighted linear regression using sklearn (no cross-validation, no regularization).

    This is a faster alternative to regress_with_kfold_rcond_cv when you don't
    need cross-validation and hyperparameter tuning. Uses standard OLS regression.

    Args:
        X: input features (N, D)
        Y: target values (N, B)
        weights: sample weights (N,)
        prefixes: optional sequence metadata for each data point

    Returns:
        RegressionResult with fitted model
    """
    # Convert to numpy
    X_np = np.asarray(X)
    Y_np = np.asarray(Y)
    weights_np = np.asarray(weights)

    # Fit standard linear regression with sample weights
    model = LinearRegression(fit_intercept=True)
    model.fit(X_np, Y_np, sample_weight=weights_np)

    # Make predictions
    Y_pred = model.predict(X_np)

    # Compute metrics using weights
    residuals = Y_pred - Y_np
    weighted_sq_residuals = (residuals ** 2) * weights_np[:, np.newaxis]
    mse = weighted_sq_residuals.mean(axis=0)
    mae = (np.abs(residuals) * weights_np[:, np.newaxis]).mean(axis=0)
    rmse = np.sqrt(mse)

    # Compute R² (weighted)
    weighted_ss_res = weighted_sq_residuals.sum()
    y_mean = np.average(Y_np, axis=0, weights=weights_np)
    weighted_ss_tot = ((Y_np - y_mean) ** 2 * weights_np[:, np.newaxis]).sum()
    r2 = 1 - (weighted_ss_res / weighted_ss_tot) if weighted_ss_tot > 0 else 0.0

    # Compute distance metric (weighted sum of Euclidean errors)
    dists = np.sqrt(np.sum(residuals ** 2, axis=1))
    dist = float((dists * weights_np).sum())

    return RegressionResult(
        best_rcond=0.0,  # No regularization used
        dist=dist,
        r2=float(r2),
        mse=mse,
        mae=mae,
        rmse=rmse,
        predictions=Y_pred,
        true_values=Y_np,
        weights=weights_np,
        per_rcond_cv_error={},  # No CV performed
        prefixes=prefixes,
    )


def project_to_simplex(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y


def regress_activations_to_beliefs(
    inputs: jax.Array,  # (batch, seq_len)
    beliefs: jax.Array,  # (batch, seq_len, B)
    probs: jax.Array,  # (batch, seq_len)
    activations_by_layer: dict[str, jax.Array],  # layer -> (batch, seq_len, d_layer)
    layer_name: str,
    rcond_values: Iterable[float],
    n_splits: int = 10,
    random_state: int = 42,
    plot_projection: bool = False,
) -> tuple[RegressionResult, go.Figure | None]:
    """Regress from activations at a given layer to beliefs using K-fold CV over rcond."""
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
    regression_results_by_step: dict[int, RegressionResult],
    title: str = "Belief Projections on 2-Simplex Over Training Steps",
) -> go.Figure:
    """Create interactive simplex projection plot with slider to navigate through training steps.

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
    regression_results_by_layer: dict[str, RegressionResult],
    title: str = "Belief Projections on 2-Simplex Across Layers",
) -> go.Figure:
    """Create interactive simplex projection plot with dropdown to switch between layers.

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
    regression_results_by_step_and_layer: dict[int, dict[str, RegressionResult]],
    title: str = "Belief Projections on 2-Simplex",
) -> go.Figure:
    """Create interactive simplex projection plot with both step slider and layer dropdown.

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

    # Note: each (step, layer) has 2 traces (true and predicted)
    # Build slider for a specific layer
    def make_slider_for_layer_regression(layer_idx: int, layer_name: str):
        slider_steps = []
        for step_idx, step in enumerate(steps_sorted):
            # Show only traces for this step and this layer
            visible_flags = []
            for s_idx in range(len(steps_sorted)):
                for l_idx in range(len(layers_sorted)):
                    # Each (step, layer) has 2 traces (true and predicted)
                    is_visible = (s_idx == step_idx and l_idx == layer_idx)
                    visible_flags.append(is_visible)
                    visible_flags.append(is_visible)

            slider_steps.append({
                "method": "update",
                "args": [
                    {"visible": visible_flags},
                    {"title": f"{title} (Step {step}, {layer_name})"},
                ],
                "label": str(step),
            })

        return {
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

    # Create dropdown that switches between layer-specific sliders
    buttons = []
    sliders_by_layer = []

    for layer_idx, layer_name in enumerate(layers_sorted):
        # Create slider for this layer
        layer_slider = make_slider_for_layer_regression(layer_idx, layer_name)
        sliders_by_layer.append(layer_slider)

        # Dropdown button switches to this layer's slider
        visible_flags = []
        for s_idx in range(len(steps_sorted)):
            for l_idx in range(len(layers_sorted)):
                # Show only traces for the first step with this layer
                is_visible = (s_idx == 0 and l_idx == layer_idx)
                visible_flags.append(is_visible)
                visible_flags.append(is_visible)

        button_dict = {
            "method": "update",
            "args": [
                {"visible": visible_flags},
                {
                    "sliders": [layer_slider],
                    "title": f"{title} (Step {steps_sorted[0]}, {layer_name})",
                },
            ],
            "label": layer_name,
        }
        buttons.append(button_dict)

    # Start with the first layer's slider
    fig.update_layout(
        sliders=[sliders_by_layer[0]],
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
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        width=600,
        height=600,
    )

    return fig