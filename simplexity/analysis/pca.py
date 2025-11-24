"""Reusable PCA helpers for activation analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.analysis.normalization import _normalize_weights

DEFAULT_VARIANCE_THRESHOLDS: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)


def compute_weighted_pca(
    x: jax.Array | np.ndarray,
    *,
    n_components: int | None = None,
    weights: jax.Array | np.ndarray | None = None,
    center: bool = True,
) -> Mapping[str, jax.Array]:
    """Compute weighted PCA for a 2D feature matrix."""
    x_arr = jnp.asarray(x)
    if x_arr.ndim != 2:
        raise ValueError("Input must be a 2D array")
    n_samples, n_features = x_arr.shape
    if n_samples == 0:
        raise ValueError("At least one sample is required")
    if n_features == 0:
        raise ValueError("At least one feature is required")

    max_rank = int(min(n_samples, n_features))
    if n_components is None:
        num_components = max_rank
    else:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_components > max_rank:
            raise ValueError(f"n_components {n_components} cannot exceed min(n_samples, n_features)={max_rank}")
        num_components = int(n_components)

    norm_weights = _normalize_weights(weights, n_samples)
    if norm_weights is None:
        mean = x_arr.mean(axis=0) if center else jnp.zeros(n_features, dtype=x_arr.dtype)
    else:
        mean = jnp.average(x_arr, axis=0, weights=norm_weights) if center else jnp.zeros(n_features, dtype=x_arr.dtype)

    x_centered = x_arr - mean
    if norm_weights is None:
        cov = (x_centered.T @ x_centered) / x_centered.shape[0]
    else:
        cov = (x_centered * norm_weights[:, None]).T @ x_centered

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_sel = eigvals[:num_components]
    eigvecs_sel = eigvecs[:, :num_components]

    total_var = eigvals.sum()
    if float(total_var) <= 0:
        explained_ratio_sel = jnp.zeros_like(eigvals_sel)
        explained_ratio_all = jnp.zeros_like(eigvals)
    else:
        explained_ratio_sel = eigvals_sel / total_var
        explained_ratio_all = eigvals / total_var

    projections = x_centered @ eigvecs_sel

    # We are using biased-covariance here, which means the absolute scale
    # of the eigenvalues is smaller by a factor of (n-1)/n compared to the
    # unbiased estimate
    return {
        "components": eigvecs_sel.T,
        "explained_variance": eigvals_sel,
        "explained_variance_ratio": explained_ratio_sel,
        "mean": mean,
        "X_proj": projections,
        "all_explained_variance": eigvals,
        "all_explained_variance_ratio": explained_ratio_all,
    }


def variance_threshold_counts(
    all_explained_variance_ratio: jax.Array,
    thresholds: Sequence[float],
) -> Mapping[float, int]:
    """Return the smallest component count reaching each variance threshold."""
    counts: dict[float, int] = {}
    cumulative = jnp.cumsum(all_explained_variance_ratio)
    for threshold in thresholds:
        idx = jnp.where(cumulative >= threshold)[0]
        counts[float(threshold)] = int(idx[0]) + 1 if len(idx) > 0 else int(cumulative.shape[0])
    return counts


def layer_pca_analysis(
    layer_activations: jax.Array,
    weights: jax.Array,
    belief_states: jax.Array | None = None,
    *,
    n_components: int | None = None,
    variance_thresholds: Sequence[float] = DEFAULT_VARIANCE_THRESHOLDS,
) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
    """Run PCA for a single layer's activations and return metrics plus projections."""
    _ = belief_states
    result = compute_weighted_pca(
        layer_activations,
        n_components=n_components,
        weights=weights,
        center=True,
    )

    cumulative_variance = jnp.cumsum(result["explained_variance_ratio"])
    scalars: dict[str, float] = {}
    for idx, value in enumerate(cumulative_variance, start=1):
        scalars[f"cumvar_{idx}"] = float(value)
    scalars["variance_explained"] = float(cumulative_variance[-1])

    threshold_counts = variance_threshold_counts(
        result["all_explained_variance_ratio"],
        variance_thresholds,
    )
    for threshold, count in threshold_counts.items():
        percentage = int(threshold * 100)
        scalars[f"n_components_{percentage}pct"] = float(count)

    projections = {"pca": result["X_proj"]}
    return scalars, projections


__all__ = [
    "DEFAULT_VARIANCE_THRESHOLDS",
    "compute_weighted_pca",
    "variance_threshold_counts",
    "layer_pca_analysis",
]
