"""Tests for PCA helper functions."""

import jax.numpy as jnp
import pytest

from simplexity.analysis.pca import (
    compute_weighted_pca,
    layer_pca_analysis,
    variance_threshold_counts,
)


def test_compute_weighted_pca_shapes() -> None:
    """Projected components should match requested dimensionality."""
    x = jnp.array([[1.0, 0.0, 2.0], [0.5, 1.5, 0.0], [1.5, 1.0, 1.0]])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = compute_weighted_pca(x, n_components=2, weights=weights)
    assert result["X_proj"].shape == (3, 2)
    assert result["components"].shape == (2, 3)


def test_variance_threshold_counts_increasing() -> None:
    """Threshold helper should return non-decreasing component counts."""
    ratios = jnp.array([0.5, 0.3, 0.2])
    thresholds = (0.5, 0.8)
    counts = variance_threshold_counts(ratios, thresholds)
    assert counts[0.8] >= counts[0.5]


def test_layer_pca_analysis_metrics() -> None:
    """Layer PCA wrapper returns metrics and projections without beliefs."""
    activations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = jnp.ones(3) / 3.0
    scalars, projections = layer_pca_analysis(
        activations,
        weights,
        belief_states=None,
        n_components=2,
        variance_thresholds=(0.5,),
    )
    assert "cumvar_1" in scalars
    assert "n_components_50pct" in scalars
    assert "pca" in projections
    assert projections["pca"].shape == (3, 2)


def test_compute_weighted_pca_rejects_bad_weights_shape() -> None:
    """Weights with mismatched shapes should raise immediately."""
    x = jnp.ones((3, 2))
    weights = jnp.ones(2)
    with pytest.raises(ValueError, match="Weights must be shape"):
        compute_weighted_pca(x, weights=weights)


def test_compute_weighted_pca_rejects_negative_weights() -> None:
    """Negative weights should be rejected during normalization."""
    x = jnp.ones((3, 2))
    weights = jnp.array([0.5, -0.1, 0.6])
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        compute_weighted_pca(x, weights=weights)


def test_compute_weighted_pca_rejects_zero_sum_weights() -> None:
    """Normalization fails gracefully when the sum of weights is zero."""
    x = jnp.ones((2, 2))
    weights = jnp.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        compute_weighted_pca(x, weights=weights)


def test_compute_weighted_pca_rejects_excess_components() -> None:
    """Requesting too many components should raise a descriptive error."""
    x = jnp.ones((2, 3))
    with pytest.raises(ValueError, match="cannot exceed min"):
        compute_weighted_pca(x, n_components=5)


def test_compute_weighted_pca_handles_zero_variance_data() -> None:
    """Constant inputs should yield zero explained variance ratios."""
    x = jnp.ones((3, 2))
    result = compute_weighted_pca(x)
    assert jnp.all(result["explained_variance_ratio"] == 0.0)
    assert jnp.all(result["all_explained_variance_ratio"] == 0.0)


def test_variance_threshold_counts_defaults_to_full_dimension() -> None:
    """When variance never exceeds the threshold, use the maximum dimension."""
    ratios = jnp.zeros(3)
    counts = variance_threshold_counts(ratios, (0.5,))
    assert counts[0.5] == 3


def test_layer_pca_analysis_zero_variance_threshold_reporting() -> None:
    """Layer PCA should propagate fallback counts into the scalar outputs."""
    activations = jnp.ones((4, 3))
    weights = jnp.ones(4) / 4.0
    scalars, projections = layer_pca_analysis(
        activations,
        weights,
        belief_states=None,
        variance_thresholds=(0.5,),
    )
    assert scalars["n_components_50pct"] == 3.0
    assert projections["pca"].shape == (4, 3)


def test_compute_weighted_pca_requires_two_dimensional_inputs() -> None:
    """PCA helper should enforce 2D inputs and non-empty shapes."""
    with pytest.raises(ValueError, match="Input must be a 2D array"):
        compute_weighted_pca(jnp.ones(3))

    with pytest.raises(ValueError, match="Cannot compute PCA on empty data"):
        compute_weighted_pca(jnp.empty((0, 3)))

    with pytest.raises(ValueError, match="Cannot compute PCA on empty data"):
        compute_weighted_pca(jnp.empty((3, 0)))


def test_compute_weighted_pca_requires_positive_component_count() -> None:
    """n_components must be positive when specified explicitly."""
    x = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="n_components must be positive"):
        compute_weighted_pca(x, n_components=0)
