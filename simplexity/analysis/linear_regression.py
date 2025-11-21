"""Reusable linear regression utilities for activation analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _standardize_features(x: Any) -> jax.Array:
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.ndim != 2:
        raise ValueError("Features must be a 2D array")
    return x_arr


def _standardize_targets(y: Any) -> jax.Array:
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    if y_arr.ndim != 2:
        raise ValueError("Targets must be a 2D array")
    return y_arr


def _standardize_weights(weights: Any, n_samples: int) -> jax.Array:
    if n_samples <= 0:
        raise ValueError("At least one sample is required")
    if weights is None:
        return jnp.ones(n_samples, dtype=jnp.float32) / float(n_samples)
    w_arr = jnp.asarray(weights, dtype=jnp.float32)
    if w_arr.ndim != 1 or w_arr.shape[0] != n_samples:
        raise ValueError("Weights must be shape (n_samples,)")
    if jnp.any(w_arr < 0):
        raise ValueError("Weights must be non-negative")
    total = float(jnp.sum(w_arr))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return w_arr / total


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
    r2 = 1.0 - (weighted_ss_res / float(weighted_ss_tot)) if weighted_ss_tot > 0 else 0.0
    dists = jnp.sqrt(jnp.sum(residuals**2, axis=1))
    dist = float(jnp.sum(dists * weights))
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
    x_arr = _standardize_features(x)
    y_arr = _standardize_targets(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Features and targets must share the same first dimension")
    w_arr = _standardize_weights(weights, x_arr.shape[0])
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
    x_arr = _standardize_features(x)
    y_arr = _standardize_targets(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Features and targets must share the same first dimension")
    w_arr = _standardize_weights(weights, x_arr.shape[0])
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
    x_arr = jnp.asarray(layer_activations)
    y_arr = jnp.asarray(belief_states)
    w_arr = jnp.asarray(weights)
    return linear_regression(x_arr, y_arr, w_arr, **kwargs)


def layer_linear_regression_svd(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | None,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Layer-wise regression helper that wraps :func:`linear_regression_svd`."""
    if belief_states is None:
        raise ValueError("linear_regression_svd requires belief_states")
    x_arr = jnp.asarray(layer_activations)
    y_arr = jnp.asarray(belief_states)
    w_arr = jnp.asarray(weights)
    return linear_regression_svd(x_arr, y_arr, w_arr, **kwargs)
