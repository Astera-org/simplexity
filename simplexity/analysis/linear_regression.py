"""Reusable linear regression utilities for activation analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.analysis.normalization import normalize_weights, standardize_features, standardize_targets


def _design_matrix(x: jax.Array, fit_intercept: bool) -> jax.Array:
    if fit_intercept:
        ones = jnp.ones((x.shape[0], 1), dtype=x.dtype)
        return jnp.concatenate([ones, x], axis=1)
    return x


def _regression_metrics(
    predictions: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
) -> Mapping[str, float]:
    residuals = predictions - targets
    weighted_sq_residuals = residuals**2 * weights[:, None]
    mse = jnp.sum(weighted_sq_residuals, axis=0)
    rmse = jnp.sqrt(mse)
    mae = jnp.sum(jnp.abs(residuals) * weights[:, None], axis=0)
    weighted_ss_res = float(weighted_sq_residuals.sum())
    target_mean = jnp.sum(targets * weights[:, None], axis=0)
    weighted_ss_tot = jnp.sum((targets - target_mean) ** 2 * weights[:, None])
    r2 = 1.0 - (weighted_ss_res / float(weighted_ss_tot)) if float(weighted_ss_tot) > 0 else 0.0
    dists = jnp.sqrt(jnp.sum(residuals**2, axis=1))
    dist = float(jnp.sum(dists * weights))
    # RMSE and MAE are returned as means over target dimensions
    # rather than sums to keep consistent with R²
    return {
        "r2": float(r2),
        "rmse": float(rmse.mean()),
        "mae": float(mae.mean()),
        "dist": dist,
    }


def linear_regression(
    x: Any,
    y: Any,
    weights: jax.Array | np.ndarray | None,
    *,
    fit_intercept: bool = True,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Weighted linear regression using a closed-form least squares solution."""
    x_arr = standardize_features(x)
    y_arr = standardize_targets(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Features and targets must share the same first dimension")
    if x_arr.shape[0] == 0:
        raise ValueError("At least one sample is required")
    w_arr = normalize_weights(weights, x_arr.shape[0])
    if w_arr is None:
        w_arr = jnp.ones(x_arr.shape[0], dtype=x_arr.dtype) / x_arr.shape[0]
    design = _design_matrix(x_arr, fit_intercept)
    sqrt_w = jnp.sqrt(w_arr)[:, None]
    weighted_design = design * sqrt_w
    weighted_targets = y_arr * sqrt_w
    beta, _, _, _ = jnp.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
    predictions = design @ beta
    scalars = _regression_metrics(predictions, y_arr, w_arr)
    projections = {"projected": predictions}
    return scalars, projections


def _compute_beta_from_svd(
    u: jax.Array,
    s: jax.Array,
    vh: jax.Array,
    weighted_targets: jax.Array,
    threshold: float,
) -> jax.Array:
    if s.size == 0:
        return jnp.zeros((vh.shape[1], weighted_targets.shape[1]), dtype=weighted_targets.dtype)
    s_inv = jnp.where(s > threshold, 1.0 / s, 0.0)
    return vh.T @ (s_inv[:, None] * (u.T @ weighted_targets))


def linear_regression_svd(
    x: Any,
    y: Any,
    weights: jax.Array | np.ndarray | None,
    *,
    rcond_values: Sequence[float] | None = None,
    fit_intercept: bool = True,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Weighted linear regression solved via SVD with configurable rcond search."""
    x_arr = standardize_features(x)
    y_arr = standardize_targets(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Features and targets must share the same first dimension")
    if x_arr.shape[0] == 0:
        raise ValueError("At least one sample is required")
    w_arr = normalize_weights(weights, x_arr.shape[0])
    if w_arr is None:
        w_arr = jnp.ones(x_arr.shape[0], dtype=x_arr.dtype) / x_arr.shape[0]
    design = _design_matrix(x_arr, fit_intercept)
    sqrt_w = jnp.sqrt(w_arr)[:, None]
    weighted_design = design * sqrt_w
    weighted_targets = y_arr * sqrt_w
    u, s, vh = jnp.linalg.svd(weighted_design, full_matrices=False)
    max_singular = float(s[0]) if s.size else 0.0
    rconds = tuple(rcond_values) if rcond_values else (1e-15,)
    best_pred: jax.Array | None = None
    best_scalars: Mapping[str, float] | None = None
    best_rcond = rconds[0]
    best_error = float("inf")
    for rcond in rconds:
        threshold = rcond * max_singular
        beta = _compute_beta_from_svd(u, s, vh, weighted_targets, threshold)
        predictions = design @ beta
        scalars = _regression_metrics(predictions, y_arr, w_arr)
        residuals = predictions - y_arr
        errors = jnp.sqrt(jnp.sum(residuals**2, axis=1))
        weighted_error = float(jnp.sum(errors * w_arr))
        if best_pred is None or weighted_error < best_error:
            best_error = weighted_error
            best_pred = predictions
            best_scalars = scalars
            best_rcond = rcond
    if best_pred is None or best_scalars is None:
        raise RuntimeError("Unable to compute linear regression solution")
    scalars = dict(best_scalars)
    scalars["best_rcond"] = float(best_rcond)
    projections = {"projected": best_pred}
    return scalars, projections


def layer_linear_regression(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | None,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Layer-wise regression helper that wraps :func:`linear_regression`."""
    if belief_states is None:
        raise ValueError("linear_regression requires belief_states")
    return linear_regression(layer_activations, belief_states, weights, **kwargs)


def layer_linear_regression_svd(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | None,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Layer-wise regression helper that wraps :func:`linear_regression_svd`."""
    if belief_states is None:
        raise ValueError("linear_regression_svd requires belief_states")
    return linear_regression_svd(layer_activations, belief_states, weights, **kwargs)
