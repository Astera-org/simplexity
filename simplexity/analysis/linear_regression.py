"""Reusable linear regression utilities for activation analysis."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.analysis.normalization import normalize_weights, standardize_features, standardize_targets
from simplexity.logger import SIMPLEXITY_LOGGER


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
    x: jax.Array,
    y: jax.Array,
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

    # Separate intercept and coefficients
    if fit_intercept:
        arrays = {
            "projected": predictions,
            "coeffs": beta[1:],  # Linear coefficients (excluding intercept)
            "intercept": beta[0:1],  # Intercept term (keep 2D: [1, n_targets])
        }
    else:
        arrays = {
            "projected": predictions,
            "coeffs": beta,  # All parameters are coefficients when no intercept
        }

    return scalars, arrays


def _compute_regression_metrics(
    x: jax.Array,
    y: jax.Array,
    weights: jax.Array | np.ndarray | None,
    beta: jax.Array,
    predictions: jax.Array | None = None,
    *,
    fit_intercept: bool = True,
):
    x_arr = standardize_features(x)
    y_arr = standardize_targets(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Features and targets must share the same first dimension")
    if x_arr.shape[0] == 0:
        raise ValueError("At least one sample is required")
    w_arr = normalize_weights(weights, x_arr.shape[0])
    if w_arr is None:
        w_arr = jnp.ones(x_arr.shape[0], dtype=x_arr.dtype) / x_arr.shape[0]
    if predictions is None:
        design = _design_matrix(x_arr, fit_intercept)
        predictions = design @ beta
    scalars = _regression_metrics(predictions, y_arr, w_arr)
    return scalars


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
    x: jax.Array,
    y: jax.Array,
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
    best_beta: jax.Array | None = None
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
            best_beta = beta
    if best_pred is None or best_scalars is None or best_beta is None:
        raise RuntimeError("Unable to compute linear regression solution")
    scalars = dict(best_scalars)
    scalars["best_rcond"] = float(best_rcond)

    # Separate intercept and coefficients
    if fit_intercept:
        arrays = {
            "projected": best_pred,
            "coeffs": best_beta[1:],  # Linear coefficients (excluding intercept)
            "intercept": best_beta[0:1],  # Intercept term (keep 2D: [1, n_targets])
        }
    else:
        arrays = {
            "projected": best_pred,
            "coeffs": best_beta,  # All parameters are coefficients when no intercept
        }

    return scalars, arrays


def _process_individual_factors(
    layer_activations: jax.Array,
    belief_states: tuple[jax.Array, ...],
    weights: jax.Array,
    use_svd: bool,
    **kwargs: Any,
) -> list[tuple[Mapping[str, float], Mapping[str, jax.Array]]]:
    """Process each factor individually using either standard or SVD regression."""
    results = []
    regression_fn = linear_regression_svd if use_svd else linear_regression
    for factor in belief_states:
        if not isinstance(factor, jax.Array):
            raise ValueError("Each factor in belief_states must be a jax.Array")
        factor_scalars, factor_arrays = regression_fn(layer_activations, factor, weights, **kwargs)
        results.append((factor_scalars, factor_arrays))
    return results


def _merge_results_with_prefix(
    scalars: dict[str, float],
    arrays: dict[str, jax.Array],
    results: tuple[Mapping[str, float], Mapping[str, jax.Array]],
    prefix: str,
) -> None:
    results_scalars, results_arrays = results
    scalars.update({f"{prefix}/{key}": value for key, value in results_scalars.items()})
    arrays.update({f"{prefix}/{key}": value for key, value in results_arrays.items()})


def _split_concat_results(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: tuple[jax.Array, ...],
    concat_results: tuple[Mapping[str, float], Mapping[str, jax.Array]],
    **kwargs: Any,
) -> list[tuple[Mapping[str, float], Mapping[str, jax.Array]]]:
    """Split concatenated regression results into individual factors."""
    _, concat_arrays = concat_results

    # Split the concatenated coefficients and projections into the individual factors
    factor_dims = [factor.shape[-1] for factor in belief_states]
    split_indices = jnp.cumsum(jnp.array(factor_dims))[:-1]

    coeffs_list = jnp.split(concat_arrays["coeffs"], split_indices, axis=-1)
    projections_list = jnp.split(concat_arrays["projected"], split_indices, axis=-1)

    # Handle intercept - split if present
    if "intercept" in concat_arrays:
        intercepts_list = jnp.split(concat_arrays["intercept"], split_indices, axis=-1)
    else:
        intercepts_list = [None] * len(belief_states)

    # Only recompute scalar metrics, reuse projections and coefficients
    # Filter out rcond_values from kwargs (only relevant for SVD during fitting, not metrics)
    metrics_kwargs = {k: v for k, v in kwargs.items() if k != "rcond_values"}
    fit_intercept = kwargs.get("fit_intercept", True)

    results = []
    for factor, coeffs, intercept, projections in zip(belief_states, coeffs_list, intercepts_list, projections_list):
        # Reconstruct full beta for metrics computation
        if intercept is not None:
            beta = jnp.concatenate([intercept, coeffs], axis=0)
        else:
            beta = coeffs

        factor_scalars = _compute_regression_metrics(
            layer_activations,
            factor,
            weights,
            beta,
            predictions=projections,
            **metrics_kwargs,
        )

        # Build factor arrays - include intercept only if present
        factor_arrays = {"projected": projections, "coeffs": coeffs}
        if intercept is not None:
            factor_arrays["intercept"] = intercept

        results.append((factor_scalars, factor_arrays))
    return results


def _compute_subspace_orthogonality(
    coeffs_pair: list[jax.Array],
) -> tuple[dict[str, float], dict[str, jax.Array]]:
    """
    Compute orthogonality metrics between two coefficient subspaces.

    Args:
        coeffs_pair: List of two coefficient matrices (excludes intercept)
    """
    # Compute the orthonormal bases for the two subspaces using QR decomposition
    q1, _ = jnp.linalg.qr(coeffs_pair[0])
    q2, _ = jnp.linalg.qr(coeffs_pair[1])
    # Compute the singular values of the interaction matrix
    interaction_matrix = q1.T @ q2
    singular_values = jnp.linalg.svd(interaction_matrix, compute_uv=False)

    # Clip the singular values to the range [0, 1]
    singular_values = jnp.clip(singular_values, 0, 1)

    # Compute the subspace overlap score
    min_dim = min(q1.shape[1], q2.shape[1])
    subspace_overlap_score = jnp.sum(singular_values**2) / min_dim

    # Compute the max singular value
    max_singular_value = jnp.max(singular_values)

    # Compute the min singular value
    min_singular_value = jnp.min(singular_values)

    # Compute the participation ratio
    participation_ratio = jnp.sum(singular_values**2) ** 2 / jnp.sum(singular_values**4)

    # Compute the entropy
    probs = singular_values**2 / jnp.sum(singular_values**2)
    entropy = -jnp.sum(probs * jnp.log(probs))

    # Compute the effective rank
    effective_rank = jnp.exp(entropy)

    scalars = {
        "subspace_overlap": float(subspace_overlap_score),
        "max_singular_value": float(max_singular_value),
        "min_singular_value": float(min_singular_value),
        "participation_ratio": float(participation_ratio),
        "entropy": float(entropy),
        "effective_rank": float(effective_rank),
    }

    singular_values = {
        "singular_values": singular_values,
    }

    return scalars, singular_values


def _compute_all_pairwise_orthogonality(
    coeffs_list: list[jax.Array],
) -> tuple[dict[str, float], dict[str, jax.Array]]:
    """
    Compute pairwise orthogonality metrics for all factor pairs.

    Args:
        coeffs_list: List of coefficient matrices (one per factor, excludes intercepts)
    """
    scalars = {}
    singular_values = {}
    factor_pairs = list(itertools.combinations(range(len(coeffs_list)), 2))
    for i, j in factor_pairs:
        coeffs_pair = [
            coeffs_list[i],
            coeffs_list[j],
        ]
        orthogonality_scalars, orthogonality_singular_values = _compute_subspace_orthogonality(coeffs_pair)
        scalars.update({f"orthogonality_{i}_{j}/{key}": value for key, value in orthogonality_scalars.items()})
        singular_values.update(
            {f"orthogonality_{i}_{j}/{key}": value for key, value in orthogonality_singular_values.items()}
        )
    return scalars, singular_values


def _handle_factored_regression(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: tuple[jax.Array, ...],
    concat_belief_states: bool,
    compute_subspace_orthogonality: bool,
    use_svd: bool,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Handle regression for factored belief states using either standard or SVD method."""
    scalars: dict[str, float] = {}
    arrays: dict[str, jax.Array] = {}

    regression_fn = linear_regression_svd if use_svd else linear_regression

    # Process concatenated belief states if requested
    if concat_belief_states:
        belief_states_concat = jnp.concatenate(belief_states, axis=-1)
        concat_results = regression_fn(layer_activations, belief_states_concat, weights, **kwargs)
        _merge_results_with_prefix(scalars, arrays, concat_results, "concat")

        # Split the concatenated parameters and projections into the individual factors
        factor_results = _split_concat_results(
            layer_activations,
            weights,
            belief_states,
            concat_results,
            **kwargs,
        )
    else:
        factor_results = _process_individual_factors(layer_activations, belief_states, weights, use_svd, **kwargs)

    for factor_idx, factor_result in enumerate(factor_results):
        _merge_results_with_prefix(scalars, arrays, factor_result, f"factor_{factor_idx}")

    if compute_subspace_orthogonality:
        # Extract coefficients (excludes intercept) for orthogonality computation
        coeffs_list = [factor_arrays["coeffs"] for _, factor_arrays in factor_results]
        orthogonality_scalars, orthogonality_singular_values = _compute_all_pairwise_orthogonality(coeffs_list)
        scalars.update(orthogonality_scalars)
        arrays.update(orthogonality_singular_values)

    return scalars, arrays


def layer_linear_regression(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | tuple[jax.Array, ...] | None,
    concat_belief_states: bool = False,
    compute_subspace_orthogonality: bool = False,
    use_svd: bool = False,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """
    Layer-wise regression helper that wraps :func:`linear_regression` or :func:`linear_regression_svd`.

    Args:
        layer_activations: Neural network activations for a single layer
        weights: Sample weights for weighted regression
        belief_states: Target belief states (single array or tuple for factored processes)
        concat_belief_states: If True and belief_states is a tuple, concatenate and regress jointly
        compute_subspace_orthogonality: If True, compute orthogonality between factor subspaces
        use_svd: If True, use SVD-based regression instead of standard least squares
        **kwargs: Additional arguments passed to regression function (fit_intercept, rcond_values, etc.)

    Returns:
        scalars: Dictionary of scalar metrics
        arrays: Dictionary of arrays (projected predictions, parameters, singular values if orthogonality computed)
    """
    if belief_states is None:
        raise ValueError("linear_regression requires belief_states")

    regression_fn = linear_regression_svd if use_svd else linear_regression

    if not isinstance(belief_states, tuple):
        if compute_subspace_orthogonality:
            SIMPLEXITY_LOGGER.warning("Subspace orthogonality cannot be computed for a single belief state")
        scalars, arrays = regression_fn(layer_activations, belief_states, weights, **kwargs)
        return scalars, arrays

    return _handle_factored_regression(
        layer_activations,
        weights,
        belief_states,
        concat_belief_states,
        compute_subspace_orthogonality,
        use_svd,
        **kwargs,
    )


def layer_linear_regression_svd(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | tuple[jax.Array, ...] | None,
    concat_belief_states: bool = False,
    compute_subspace_orthogonality: bool = False,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """
    Layer-wise SVD regression helper (wrapper around layer_linear_regression with use_svd=True).

    This function is provided for backward compatibility and convenience.
    Consider using layer_linear_regression with use_svd=True for new code.
    """
    return layer_linear_regression(
        layer_activations=layer_activations,
        weights=weights,
        belief_states=belief_states,
        concat_belief_states=concat_belief_states,
        compute_subspace_orthogonality=compute_subspace_orthogonality,
        use_svd=True,
        **kwargs,
    )
