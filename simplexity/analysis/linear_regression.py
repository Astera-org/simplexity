"""Reusable linear regression utilities for activation analysis."""

# pylint: disable=all # Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all # Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import callback

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

    if fit_intercept:
        arrays = {
            "projected": predictions,
            "targets": y_arr,
            "coeffs": beta[1:],
            "intercept": beta[:1],
        }
    else:
        arrays = {
            "projected": predictions,
            "targets": y_arr,
            "coeffs": beta,
        }

    return scalars, arrays


def _compute_regression_metrics(  # pylint: disable=too-many-arguments
    x: jax.Array,
    y: jax.Array,
    weights: jax.Array | np.ndarray | None,
    beta: jax.Array,
    predictions: jax.Array | None = None,
    *,
    fit_intercept: bool = True,
) -> Mapping[str, float]:
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
    assert best_pred is not None
    assert best_scalars is not None
    assert best_beta is not None
    scalars = dict(best_scalars)
    scalars["best_rcond"] = float(best_rcond)

    if fit_intercept:
        arrays = {
            "projected": best_pred,
            "targets": y_arr,
            "coeffs": best_beta[1:],
            "intercept": best_beta[:1],
        }
    else:
        arrays = {
            "projected": best_pred,
            "targets": y_arr,
            "coeffs": best_beta,
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


def _merge_results_with_suffix(
    scalars: dict[str, float],
    arrays: dict[str, jax.Array],
    results: tuple[Mapping[str, float], Mapping[str, jax.Array]],
    suffix: str,
) -> None:
    results_scalars, results_arrays = results
    scalars.update({f"{key}/{suffix}": value for key, value in results_scalars.items()})
    arrays.update({f"{key}/{suffix}": value for key, value in results_arrays.items()})


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
    targets_list = jnp.split(concat_arrays["targets"], split_indices, axis=-1)

    # Handle intercept - split if present
    if "intercept" in concat_arrays:
        intercepts_list = jnp.split(concat_arrays["intercept"], split_indices, axis=-1)
    else:
        intercepts_list = [None] * len(belief_states)

    # Only recompute scalar metrics, reuse projections and coefficients
    # Filter out rcond_values from kwargs (only relevant for SVD during fitting, not metrics)
    metrics_kwargs = {k: v for k, v in kwargs.items() if k != "rcond_values"}

    results = []
    for factor, coeffs, intercept, projections, targets in zip(
        belief_states, coeffs_list, intercepts_list, projections_list, targets_list, strict=True
    ):
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
        factor_arrays = {"projected": projections, "targets": targets, "coeffs": coeffs}
        if intercept is not None:
            factor_arrays["intercept"] = intercept

        results.append((factor_scalars, factor_arrays))
    return results


def get_robust_basis(matrix: jax.Array) -> jax.Array:
    """Extracts an orthonormal basis for the column space of the matrix.

    Handles rank deficiency gracefully by discarding directions associated with singular values below a
    certain tolerance.
    """
    u, s, _ = jnp.linalg.svd(matrix, full_matrices=False)

    max_dim = max(matrix.shape)
    eps = jnp.finfo(matrix.dtype).eps
    tol = s[0] * max_dim * eps

    valid_dims = s > tol
    basis = u[:, valid_dims]
    return basis


def _compute_subspace_orthogonality(
    basis_pair: list[jax.Array],
) -> tuple[dict[str, float], dict[str, jax.Array]]:
    """Compute orthogonality metrics between two coefficient subspaces.

    Returns dict keys: overlap, sv_max, sv_min, p_ratio, entropy, eff_rank, singular_values.
    """
    q1 = basis_pair[0]
    q2 = basis_pair[1]

    # Compute the singular values of the interaction matrix
    interaction_matrix = q1.T @ q2
    singular_values = jnp.linalg.svd(interaction_matrix, compute_uv=False)
    singular_values = jnp.clip(singular_values, 0, 1)

    # Compute the subspace overlap score
    min_dim = min(q1.shape[1], q2.shape[1])
    sum_sq_sv = jnp.sum(singular_values**2)
    sum_quad_sv = jnp.sum(singular_values**4)

    is_degenerate = sum_quad_sv == 0

    def _warn_subspace_issues(sum_quad: jax.Array, num_zero_probs: jax.Array) -> None:
        if sum_quad.item() == 0:
            SIMPLEXITY_LOGGER.warning(
                "Degenerate subspace detected during orthogonality computation."
                " All singular values are zero."
                " Setting probability values and participation ratio to zero."
            )
        if num_zero_probs.item() > 0:
            SIMPLEXITY_LOGGER.warning(
                "Encountered %d probability values of zero during entropy computation."
                " This is likely due to numerical instability."
                " Setting corresponding entropy contribution to zero.",
                num_zero_probs.item(),
            )

    pratio_denominator_safe = jnp.where(is_degenerate, 1.0, sum_quad_sv)
    probs_denominator_safe = jnp.where(is_degenerate, 1.0, sum_sq_sv)
    participation_ratio = sum_sq_sv**2 / pratio_denominator_safe

    subspace_overlap_score = sum_sq_sv / min_dim

    probs = singular_values**2 / probs_denominator_safe
    num_zeros = jnp.sum(probs == 0)

    callback(_warn_subspace_issues, sum_quad_sv, num_zeros)

    p_log_p = probs * jnp.log(probs)
    entropy = -jnp.sum(jnp.where(probs > 0, p_log_p, 0.0))

    # Compute the effective rank
    effective_rank = jnp.exp(entropy)

    scalars = {
        "overlap": float(subspace_overlap_score),
        "sv_max": float(jnp.max(singular_values)),
        "sv_min": float(jnp.min(singular_values)),
        "p_ratio": float(participation_ratio),
        "entropy": float(entropy),
        "eff_rank": float(effective_rank),
    }

    arrays = {
        "singular_values": singular_values,
    }

    return scalars, arrays


def _compute_all_pairwise_orthogonality(
    coeffs_list: list[jax.Array],
) -> tuple[dict[str, float], dict[str, jax.Array]]:
    """Compute pairwise orthogonality metrics for all factor pairs."""
    scalars = {}
    arrays = {}
    factor_pairs = list(itertools.combinations(range(len(coeffs_list)), 2))
    basis_list = [get_robust_basis(coeffs) for coeffs in coeffs_list]  # computes orthonormal basis of coeff matrix
    for i, j in factor_pairs:
        basis_pair = [basis_list[i], basis_list[j]]
        orthogonality_scalars, orthogonality_arrays = _compute_subspace_orthogonality(basis_pair)
        scalars.update({f"{i},{j}/{key}": value for key, value in orthogonality_scalars.items()})
        arrays.update({f"{i},{j}/{key}": value for key, value in orthogonality_arrays.items()})
    return scalars, arrays


def _handle_factored_regression(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: tuple[jax.Array, ...],
    concat_belief_states: bool,
    compute_subspace_orthogonality: bool,
    use_svd: bool,
    **kwargs: Any,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Handle regression for two or more factored belief states using either standard or SVD method."""
    if len(belief_states) < 2:
        raise ValueError("At least two factors are required for factored regression")

    scalars: dict[str, float] = {}
    arrays: dict[str, jax.Array] = {}

    regression_fn = linear_regression_svd if use_svd else linear_regression

    # Process concatenated belief states if requested
    if concat_belief_states:
        belief_states_concat = jnp.concatenate(belief_states, axis=-1)
        concat_results = regression_fn(layer_activations, belief_states_concat, weights, **kwargs)
        _merge_results_with_suffix(scalars, arrays, concat_results, "Fcat")

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
        _merge_results_with_suffix(scalars, arrays, factor_result, f"F{factor_idx}")

    if compute_subspace_orthogonality:
        # Extract coefficients (excludes intercept) for orthogonality computation
        coeffs_list = [factor_arrays["coeffs"] for _, factor_arrays in factor_results]
        orthogonality_scalars, orthogonality_arrays = _compute_all_pairwise_orthogonality(coeffs_list)
        for key, value in orthogonality_scalars.items():
            factors, metric = key.split("/")
            new_key = f"orth/{metric}/F{factors}"
            scalars.update({new_key: value})
        for key, value in orthogonality_arrays.items():
            factors, metric = key.split("/")
            new_key = f"orth/{metric}/F{factors}"
            arrays.update({new_key: value})
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
    """Layer-wise regression helper that wraps linear_regression or linear_regression_svd."""
    # If no belief states are provided, raise an error
    if (
        belief_states is None
        or (isinstance(belief_states, tuple) and len(belief_states) == 0)
        or (isinstance(belief_states, jax.Array) and belief_states.size == 0)
    ):
        raise ValueError("linear_regression requires belief_states")

    regression_fn = linear_regression_svd if use_svd else linear_regression

    if not isinstance(belief_states, tuple) or len(belief_states) == 1:
        if compute_subspace_orthogonality:
            SIMPLEXITY_LOGGER.warning(
                "Subspace orthogonality requires multiple factors."
                " Received single factor of type %s; skipping orthogonality metrics.",
                type(belief_states).__name__,
            )
        belief_states = belief_states[0] if isinstance(belief_states, tuple) else belief_states
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
