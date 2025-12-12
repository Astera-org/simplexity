"""Tests for reusable linear regression helpers."""

# pylint: disable=too-many-lines

import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.analysis.linear_regression import (
    layer_linear_regression,
    layer_linear_regression_svd,
    linear_regression,
    linear_regression_svd,
)


def _compute_orthogonality_threshold(
    x: jax.Array,
    *factors: jax.Array,
    safety_factor: int = 10,
) -> float:
    """Compute principled threshold for near-zero orthogonality checks.

    Threshold is based on machine precision scaled by problem dimensions.
    For orthogonality via QR + SVD, typical numerical error is O(ε·n) where
    ε is machine epsilon and n is the maximum relevant dimension.

    Args:
        x: Input features array (used for dtype and dimension)
        *factors: Factor arrays being compared (used for output dimensions)
        safety_factor: Multiplicative safety factor (default 10)

    Returns:
        Threshold value for considering singular values as effectively zero
    """
    eps = jnp.finfo(x.dtype).eps
    n_features = x.shape[1]
    factor_dims = [f.shape[1] for f in factors]
    max_dim = max(n_features, *factor_dims)
    return float(max_dim * eps * safety_factor)


def test_linear_regression_perfect_fit() -> None:
    """Verify weighted least squares recovers a perfect linear relation."""
    x = jnp.arange(6.0).reshape(-1, 1)
    y = 3.0 * x + 2.0
    weights = jnp.ones(x.shape[0])

    scalars, arrays = linear_regression(x, y, weights)

    assert pytest.approx(1.0) == scalars["r2"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["rmse"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["mae"]
    chex.assert_trees_all_close(arrays["projected"], y)


def test_linear_regression_svd_selects_best_rcond() -> None:
    """Ensure the SVD variant exposes chosen rcond and predictions."""
    x = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]])
    y = jnp.sum(x, axis=1, keepdims=True)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])

    scalars, arrays = linear_regression_svd(
        x,
        y,
        weights,
        rcond_values=[1e-6, 1e-4, 1e-2],
    )

    assert scalars["best_rcond"] in {1e-6, 1e-4, 1e-2}
    chex.assert_trees_all_close(arrays["projected"], y)


def test_layer_regression_requires_targets() -> None:
    """Layer helpers surface missing belief state errors."""
    x = jnp.ones((3, 2))
    weights = jnp.ones(3)

    with pytest.raises(ValueError, match="requires belief_states"):
        layer_linear_regression(x, weights, None)

    with pytest.raises(ValueError, match="requires belief_states"):
        layer_linear_regression_svd(x, weights, None)


def test_linear_regression_rejects_mismatched_weights() -> None:
    """Weights must align with the sample dimension."""
    x = jnp.ones((4, 1))
    y = jnp.ones((4, 1))
    weights = jnp.ones(3)

    with pytest.raises(ValueError, match="Weights must be shape"):
        linear_regression(x, y, weights)


def test_linear_regression_rejects_negative_weights() -> None:
    """Negative weights should be rejected before fitting."""
    x = jnp.ones((4, 1))
    y = jnp.ones((4, 1))
    weights = jnp.array([0.5, -0.1, 0.3, 0.3])

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        linear_regression(x, y, weights)


def test_linear_regression_rejects_zero_sum_weights() -> None:
    """Weight normalization should fail when the sum is zero."""
    x = jnp.ones((2, 1))
    y = jnp.ones((2, 1))
    weights = jnp.array([0.0, 0.0])

    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        linear_regression(x, y, weights)


def test_linear_regression_without_intercept_uses_uniform_weights() -> None:
    """When weights are None the helper should apply uniform weighting."""
    x = jnp.arange(1.0, 4.0)[:, None]
    y = 2.0 * x

    scalars, arrays = linear_regression(x, y, None, fit_intercept=False)

    assert pytest.approx(1.0) == scalars["r2"]
    chex.assert_trees_all_close(arrays["projected"], y)


def test_linear_regression_svd_handles_empty_features() -> None:
    """SVD helper should handle inputs with no feature columns."""
    x = jnp.empty((3, 0))
    y = jnp.arange(3.0)[:, None]
    weights = jnp.ones(3)

    scalars, arrays = linear_regression_svd(x, y, weights, fit_intercept=False)

    assert scalars["best_rcond"] == pytest.approx(1e-15)
    chex.assert_trees_all_close(arrays["projected"], jnp.zeros_like(y))


def test_linear_regression_accepts_one_dimensional_inputs() -> None:
    """1D features and targets should be promoted to column vectors."""
    x = jnp.arange(4.0)
    y = 5.0 * x + 1.0
    weights = jnp.ones_like(x)

    scalars, arrays = linear_regression(x, y, weights)

    assert pytest.approx(1.0) == scalars["r2"]
    chex.assert_trees_all_close(arrays["projected"], y[:, None])


def test_linear_regression_rejects_high_rank_inputs() -> None:
    """Features and targets must be 2D after standardization."""
    x = jnp.ones((2, 1, 1))
    y = jnp.ones((2, 1))
    weights = jnp.ones(2)

    with pytest.raises(ValueError, match="Features must be a 2D array"):
        linear_regression(x, y, weights)

    y_bad = jnp.ones((2, 1, 1))
    with pytest.raises(ValueError, match="Targets must be a 2D array"):
        linear_regression(jnp.ones((2, 1)), y_bad, weights)


def test_linear_regression_requires_nonempty_weighted_samples() -> None:
    """Even with empty inputs, the solver should reject missing samples."""
    x = jnp.empty((0, 1))
    y = jnp.empty((0, 1))
    weights = jnp.empty((0,))

    with pytest.raises(ValueError, match="At least one sample is required"):
        linear_regression(x, y, weights)


def test_linear_regression_mismatched_feature_target_shapes() -> None:
    """Mismatch in sample dimension should raise for both solvers."""
    x = jnp.ones((3, 1))
    y = jnp.ones((2, 1))
    weights = jnp.ones(3)

    with pytest.raises(ValueError, match="Features and targets must share the same first dimension"):
        linear_regression(x, y, weights)

    with pytest.raises(ValueError, match="Features and targets must share the same first dimension"):
        linear_regression_svd(x, y, weights)


def test_linear_regression_svd_falls_back_to_default_rcond() -> None:
    """Empty rcond lists should fall back to the default threshold search."""
    x = jnp.ones((3, 1))
    y = jnp.ones((3, 1))
    weights = jnp.ones(3)

    scalars, _ = linear_regression_svd(x, y, weights, rcond_values=[])

    assert scalars["best_rcond"] == pytest.approx(1e-15)


def test_layer_linear_regression_svd_runs_end_to_end() -> None:
    """Layer helper should proxy through to the base implementation."""
    x = jnp.arange(6.0).reshape(3, 2)
    weights = jnp.ones(3) / 3.0
    beliefs = 2.0 * x.sum(axis=1, keepdims=True)

    scalars, arrays = layer_linear_regression_svd(
        x,
        weights,
        beliefs,
        rcond_values=[1e-3],
    )

    assert pytest.approx(1.0, abs=1e-6) == scalars["r2"]
    chex.assert_trees_all_close(arrays["projected"], beliefs)


def test_layer_linear_regression_belief_states_tuple_default() -> None:
    """By default, layer regression should regress to each factor separately if given a tuple of belief states."""
    x = jnp.arange(12.0).reshape(4, 3)  # 4 samples, 3 features
    weights = jnp.ones(4) / 4.0

    # Two factors: factor 0 has 2 states, factor 1 has 3 states
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])  # [4, 2]
    factor_1 = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])  # [4, 3]
    factored_beliefs = (factor_0, factor_1)

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
    )

    # Should have separate metrics for each factor
    assert "factor_0/r2" in scalars
    assert "factor_1/r2" in scalars
    assert "factor_0/rmse" in scalars
    assert "factor_1/rmse" in scalars
    assert "factor_0/mae" in scalars
    assert "factor_1/mae" in scalars
    assert "factor_0/dist" in scalars
    assert "factor_1/dist" in scalars

    # Should have separate projections for each factor
    assert "factor_0/projected" in arrays
    assert "factor_1/projected" in arrays

    # Should have separate parameters for each factor
    assert "factor_0/coeffs" in arrays
    assert "factor_1/coeffs" in arrays

    # Should have separate intercepts for each factor by default
    assert "factor_0/intercept" in arrays
    assert "factor_1/intercept" in arrays

    # Check shapes
    assert arrays["factor_0/projected"].shape == factor_0.shape
    assert arrays["factor_1/projected"].shape == factor_1.shape
    assert arrays["factor_0/coeffs"].shape == (x.shape[1], factor_0.shape[1])
    assert arrays["factor_1/coeffs"].shape == (x.shape[1], factor_1.shape[1])
    assert arrays["factor_0/intercept"].shape == (1, factor_0.shape[1])
    assert arrays["factor_1/intercept"].shape == (1, factor_1.shape[1])


def test_layer_linear_regression_svd_belief_states_tuple_default() -> None:
    """By default, layer regression SVD should regress to each factor separately if given a tuple of belief states."""
    x = jnp.arange(12.0).reshape(4, 3)  # 4 samples, 3 features
    weights = jnp.ones(4) / 4.0

    # Two factors: factor 0 has 2 states, factor 1 has 3 states
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])  # [4, 2]
    factor_1 = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])  # [4, 3]
    factored_beliefs = (factor_0, factor_1)

    scalars, arrays = layer_linear_regression_svd(
        x,
        weights,
        factored_beliefs,
        rcond_values=[1e-6],
    )

    # Should have ALL regression metrics for each factor including best_rcond
    for factor in [0, 1]:
        assert f"factor_{factor}/r2" in scalars
        assert f"factor_{factor}/rmse" in scalars
        assert f"factor_{factor}/mae" in scalars
        assert f"factor_{factor}/dist" in scalars
        assert f"factor_{factor}/best_rcond" in scalars

    # Should have separate projections for each factor
    assert "factor_0/projected" in arrays
    assert "factor_1/projected" in arrays

    # Should have separate coefficients for each factor
    assert "factor_0/coeffs" in arrays
    assert "factor_1/coeffs" in arrays

    # Should have separate intercepts for each factor by default
    assert "factor_0/intercept" in arrays
    assert "factor_1/intercept" in arrays

    # Check shapes
    assert arrays["factor_0/projected"].shape == factor_0.shape
    assert arrays["factor_1/projected"].shape == factor_1.shape
    assert arrays["factor_0/coeffs"].shape == (x.shape[1], factor_0.shape[1])
    assert arrays["factor_1/coeffs"].shape == (x.shape[1], factor_1.shape[1])
    assert arrays["factor_0/intercept"].shape == (1, factor_0.shape[1])
    assert arrays["factor_1/intercept"].shape == (1, factor_1.shape[1])


def test_layer_linear_regression_belief_states_tuple_single_factor() -> None:
    """Single-element tuple should behave the same as passing a single array."""
    x = jnp.arange(9.0).reshape(3, 3)
    weights = jnp.ones(3) / 3.0

    # Single factor in tuple
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
    factored_beliefs = (factor_0,)

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
    )

    # Should have same structure as non-tuple case
    assert "r2" in scalars
    assert "rmse" in scalars
    assert "mae" in scalars
    assert "dist" in scalars
    assert "projected" in arrays
    assert "coeffs" in arrays
    assert "intercept" in arrays

    # Verify it matches non-tuple behavior
    scalars_non_tuple, arrays_non_tuple = layer_linear_regression(x, weights, factor_0)

    chex.assert_trees_all_close(scalars, scalars_non_tuple)
    chex.assert_trees_all_close(arrays, arrays_non_tuple)


def test_orthogonality_with_orthogonal_subspaces() -> None:
    """Orthogonal factors constructed explicitly should have near-zero overlap."""

    # Create truly orthogonal coefficient matrices by construction
    n_samples, n_features = 100, 6
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Define orthogonal coefficient matrices
    # w_0 uses first 3 features, w_1 uses last 3 features
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # (6, 2)
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (6, 2)

    # Generate factors using orthogonal subspaces (no intercept for simplicity)
    factor_0 = x @ w_0  # (100, 2)
    factor_1 = x @ w_1  # (100, 2)
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=False,  # No intercept for cleaner test
    )

    # Should have standard factor metrics with perfect fit
    assert scalars["factor_0/r2"] > 0.99  # Should fit nearly perfectly
    assert scalars["factor_1/r2"] > 0.99

    # Should have ALL orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars
    assert "orthogonality_0_1/min_singular_value" in scalars
    assert "orthogonality_0_1/participation_ratio" in scalars
    assert "orthogonality_0_1/entropy" in scalars
    assert "orthogonality_0_1/effective_rank" in scalars

    # Compute principled threshold based on machine precision and problem size
    threshold = _compute_orthogonality_threshold(x, factor_0, factor_1)

    # Should indicate near-zero overlap (orthogonal by construction)
    assert scalars["orthogonality_0_1/subspace_overlap"] < threshold
    assert scalars["orthogonality_0_1/max_singular_value"] < threshold

    # Should have singular values in arrays
    assert "orthogonality_0_1/singular_values" in arrays
    # Both factors have 2 dimensions, so min(2, 2) = 2 singular values
    assert arrays["orthogonality_0_1/singular_values"].shape[0] == 2
    # All singular values should be near zero (orthogonal)
    assert jnp.all(arrays["orthogonality_0_1/singular_values"] < threshold)


def test_orthogonality_with_aligned_subspaces() -> None:
    """Aligned factors with identical column spaces should have high overlap."""

    # Create truly aligned coefficient matrices by construction
    n_samples, n_features = 100, 6
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Define aligned coefficient matrices - w_1 = w_0 @ A for invertible A
    # This ensures span(w_1) = span(w_0)
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # (6, 2)
    w_1 = jnp.array([[0.5, 1.0], [1.0, 0.5], [1.5, 1.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # (6, 2)

    # Generate factors using aligned subspaces (no intercept for simplicity)
    factor_0 = x @ w_0  # (100, 2)
    factor_1 = x @ w_1  # (100, 2)
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=False,  # No intercept for cleaner test
    )

    # Should have standard factor metrics with perfect fit
    assert scalars["factor_0/r2"] > 0.99  # Should fit nearly perfectly
    assert scalars["factor_1/r2"] > 0.99

    # Should have ALL orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars
    assert "orthogonality_0_1/min_singular_value" in scalars
    assert "orthogonality_0_1/participation_ratio" in scalars
    assert "orthogonality_0_1/entropy" in scalars
    assert "orthogonality_0_1/effective_rank" in scalars

    # Should indicate high overlap (aligned by construction)
    assert scalars["orthogonality_0_1/subspace_overlap"] > 0.99
    assert scalars["orthogonality_0_1/max_singular_value"] > 0.99

    # Should have singular values in arrays
    assert "orthogonality_0_1/singular_values" in arrays
    # Both factors have 2 dimensions, so min(2, 2) = 2 singular values
    assert arrays["orthogonality_0_1/singular_values"].shape[0] == 2
    # All singular values should be near 1.0 (perfectly aligned)
    assert jnp.all(arrays["orthogonality_0_1/singular_values"] > 0.99)


def test_orthogonality_with_three_factors() -> None:
    """Three factors should produce all pairwise orthogonality combinations."""

    # Create three mutually orthogonal coefficient matrices
    n_samples, n_features = 100, 6
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Define three orthogonal coefficient matrices using disjoint features
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # Uses features 0-1
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # Uses features 2-3
    w_2 = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # Uses features 4-5

    # Generate factors using orthogonal subspaces
    factor_0 = x @ w_0  # (100, 2)
    factor_1 = x @ w_1  # (100, 2)
    factor_2 = x @ w_2  # (100, 2)
    factored_beliefs = (factor_0, factor_1, factor_2)
    weights = jnp.ones(n_samples) / n_samples

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=False,
    )

    # Should have standard factor metrics for all three factors
    assert scalars["factor_0/r2"] > 0.99
    assert scalars["factor_1/r2"] > 0.99
    assert scalars["factor_2/r2"] > 0.99

    # Compute principled threshold based on machine precision and problem size
    threshold = _compute_orthogonality_threshold(x, factor_0, factor_1, factor_2)

    # Should have ALL three pairwise orthogonality combinations
    pairwise_keys = ["orthogonality_0_1", "orthogonality_0_2", "orthogonality_1_2"]
    for pair_key in pairwise_keys:
        assert f"{pair_key}/subspace_overlap" in scalars
        assert f"{pair_key}/max_singular_value" in scalars
        assert f"{pair_key}/min_singular_value" in scalars
        assert f"{pair_key}/participation_ratio" in scalars
        assert f"{pair_key}/entropy" in scalars
        assert f"{pair_key}/effective_rank" in scalars
        assert f"{pair_key}/singular_values" in arrays

        # All pairs should be orthogonal (near-zero overlap)
        overlap = scalars[f"{pair_key}/subspace_overlap"]
        assert overlap < threshold, f"{pair_key} subspace_overlap={overlap} >= threshold={threshold}"

        max_sv = scalars[f"{pair_key}/max_singular_value"]
        assert max_sv < threshold, f"{pair_key} max_singular_value={max_sv} >= threshold={threshold}"

        # Each pair has 2D subspaces, so 2 singular values
        assert arrays[f"{pair_key}/singular_values"].shape[0] == 2
        svs = arrays[f"{pair_key}/singular_values"]
        assert jnp.all(svs < threshold), f"{pair_key} singular_values={svs} not all < threshold={threshold}"


def test_orthogonality_not_computed_by_default() -> None:
    """Orthogonality metrics should not be computed when compute_subspace_orthogonality=False."""

    # Setup two-factor regression
    n_samples, n_features = 50, 4
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    factor_0 = x @ w_0
    factor_1 = x @ w_1
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    # Run WITHOUT compute_subspace_orthogonality (default is False)
    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        fit_intercept=False,
    )

    # Should have standard factor metrics
    assert "factor_0/r2" in scalars
    assert "factor_1/r2" in scalars

    # Should NOT have any orthogonality metrics
    orthogonality_keys = [
        "orthogonality_0_1/subspace_overlap",
        "orthogonality_0_1/max_singular_value",
        "orthogonality_0_1/min_singular_value",
        "orthogonality_0_1/participation_ratio",
        "orthogonality_0_1/entropy",
        "orthogonality_0_1/effective_rank",
    ]
    for key in orthogonality_keys:
        assert key not in scalars

    # Should NOT have orthogonality singular values in arrays
    assert "orthogonality_0_1/singular_values" not in arrays


def test_orthogonality_warning_for_single_belief_state(caplog: pytest.LogCaptureFixture) -> None:
    """Should warn when requesting orthogonality with a single belief state."""

    # Setup single-factor regression
    n_samples, n_features = 30, 4
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))
    belief_state = jax.random.normal(key, (n_samples, 2))
    weights = jnp.ones(n_samples) / n_samples

    # Request orthogonality with single belief state (not a tuple)
    with caplog.at_level("WARNING"):
        scalars, arrays = layer_linear_regression(
            x,
            weights,
            belief_state,
            compute_subspace_orthogonality=True,
            fit_intercept=False,
        )

    # Should have logged a warning
    assert "Subspace orthogonality requires multiple factors." in caplog.text

    # Should still run regression successfully
    assert "r2" in scalars
    assert "projected" in arrays

    # Should NOT have orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" not in scalars
    assert "orthogonality_0_1/singular_values" not in arrays


def test_use_svd_flag_equivalence() -> None:
    """layer_linear_regression with use_svd=True should match layer_linear_regression_svd."""

    n_samples, n_features = 40, 4
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Test with single belief state
    belief_state = jax.random.normal(key, (n_samples, 3))
    weights = jnp.ones(n_samples) / n_samples
    rcond_values = [1e-6, 1e-4]

    # Method 1: use_svd=True
    scalars_flag, arrays_flag = layer_linear_regression(
        x,
        weights,
        belief_state,
        use_svd=True,
        rcond_values=rcond_values,
    )

    # Method 2: layer_linear_regression_svd
    scalars_wrapper, arrays_wrapper = layer_linear_regression_svd(
        x,
        weights,
        belief_state,
        rcond_values=rcond_values,
    )

    # Should produce identical results
    assert scalars_flag.keys() == scalars_wrapper.keys()
    for key, value in scalars_flag.items():
        assert value == pytest.approx(scalars_wrapper[key])

    assert arrays_flag.keys() == arrays_wrapper.keys()
    for key, value in arrays_flag.items():
        chex.assert_trees_all_close(value, arrays_wrapper[key])

    # Test with factored belief states
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    factor_0 = x @ w_0
    factor_1 = x @ w_1
    factored_beliefs = (factor_0, factor_1)

    # Method 1: use_svd=True with factored beliefs
    scalars_flag_fact, arrays_flag_fact = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        use_svd=True,
        rcond_values=rcond_values,
    )

    # Method 2: layer_linear_regression_svd with factored beliefs
    scalars_wrapper_fact, arrays_wrapper_fact = layer_linear_regression_svd(
        x,
        weights,
        factored_beliefs,
        rcond_values=rcond_values,
    )

    # Should produce identical results
    assert scalars_flag_fact.keys() == scalars_wrapper_fact.keys()
    for key, value in scalars_flag_fact.items():
        assert value == pytest.approx(scalars_wrapper_fact[key])

    assert arrays_flag_fact.keys() == arrays_wrapper_fact.keys()
    for key, value in arrays_flag_fact.items():
        chex.assert_trees_all_close(value, arrays_wrapper_fact[key])


def test_use_svd_with_orthogonality() -> None:
    """SVD regression should work with orthogonality computation."""

    n_samples, n_features = 80, 6
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Create orthogonal coefficient matrices
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    factor_0 = x @ w_0
    factor_1 = x @ w_1
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    # Run SVD regression with orthogonality computation
    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        use_svd=True,
        compute_subspace_orthogonality=True,
        rcond_values=[1e-6],
        fit_intercept=False,
    )

    # Should have standard factor metrics with SVD
    assert "factor_0/r2" in scalars
    assert "factor_1/r2" in scalars
    assert "factor_0/best_rcond" in scalars
    assert "factor_1/best_rcond" in scalars

    # Should have orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars
    assert "orthogonality_0_1/singular_values" in arrays

    # Compute principled threshold
    threshold = _compute_orthogonality_threshold(x, factor_0, factor_1)

    # Should indicate near-zero overlap (orthogonal by construction)
    assert scalars["orthogonality_0_1/subspace_overlap"] < threshold
    assert scalars["orthogonality_0_1/max_singular_value"] < threshold

    # Should have good regression fit
    assert scalars["factor_0/r2"] > 0.99
    assert scalars["factor_1/r2"] > 0.99


def test_orthogonality_with_different_subspace_dimensions() -> None:
    """Orthogonality should work when factors have different output dimensions."""

    n_samples, n_features = 100, 8
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Create orthogonal coefficient matrices with different output dimensions
    # factor_0 has 2 output dimensions, factor_1 has 5 output dimensions
    w_0 = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )  # (8, 2)
    w_1 = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )  # (8, 5)

    factor_0 = x @ w_0  # (100, 2)
    factor_1 = x @ w_1  # (100, 5)
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=False,
    )

    # Should have standard factor metrics
    assert scalars["factor_0/r2"] > 0.99
    assert scalars["factor_1/r2"] > 0.99

    # Should have orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars
    assert "orthogonality_0_1/singular_values" in arrays

    # Compute principled threshold
    threshold = _compute_orthogonality_threshold(x, factor_0, factor_1)

    # Should indicate near-zero overlap (orthogonal by construction)
    assert scalars["orthogonality_0_1/subspace_overlap"] < threshold
    assert scalars["orthogonality_0_1/max_singular_value"] < threshold

    # Singular values shape should be min(2, 5) = 2
    assert arrays["orthogonality_0_1/singular_values"].shape[0] == 2
    assert jnp.all(arrays["orthogonality_0_1/singular_values"] < threshold)


def test_orthogonality_with_contained_subspace() -> None:
    """Smaller subspace fully contained in larger subspace should show high alignment."""

    n_samples, n_features = 100, 8
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Create coefficient matrices where factor_0's subspace is contained in factor_1's
    # factor_0: 2D subspace using features [0, 1]
    # factor_1: 3D subspace using features [0, 1, 2] (contains factor_0's space)
    w_0 = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )  # (8, 2)
    w_1 = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )  # (8, 3)

    factor_0 = x @ w_0  # (100, 2)
    factor_1 = x @ w_1  # (100, 3)
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=False,
    )

    # Should have standard factor metrics
    assert scalars["factor_0/r2"] > 0.99
    assert scalars["factor_1/r2"] > 0.99

    # Should have orthogonality metrics
    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars
    assert "orthogonality_0_1/singular_values" in arrays

    # Singular values shape should be min(2, 3) = 2
    assert arrays["orthogonality_0_1/singular_values"].shape[0] == 2

    # Since factor_0's subspace is contained in factor_1's, singular values should be near 1.0
    # (indicating perfect alignment in the 2D shared subspace)
    assert scalars["orthogonality_0_1/subspace_overlap"] > 0.99
    assert scalars["orthogonality_0_1/max_singular_value"] > 0.99
    assert scalars["orthogonality_0_1/min_singular_value"] > 0.99
    assert jnp.all(arrays["orthogonality_0_1/singular_values"] > 0.99)


def test_orthogonality_excludes_intercept() -> None:
    """Orthogonality should be computed using only coefficients, not intercept."""

    n_samples, n_features = 100, 6
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n_samples, n_features))

    # Create orthogonal coefficient matrices
    w_0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    w_1 = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Add different intercepts to the factors
    intercept_0 = jnp.array([[5.0, -3.0]])
    intercept_1 = jnp.array([[10.0, 7.0]])

    factor_0 = x @ w_0 + intercept_0  # (100, 2)
    factor_1 = x @ w_1 + intercept_1  # (100, 2)
    factored_beliefs = (factor_0, factor_1)
    weights = jnp.ones(n_samples) / n_samples

    # Run with fit_intercept=True
    scalars, arrays = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        compute_subspace_orthogonality=True,
        fit_intercept=True,
    )

    # Should have intercepts for both factors
    assert "factor_0/intercept" in arrays
    assert "factor_1/intercept" in arrays

    # Should have good regression fit
    assert scalars["factor_0/r2"] > 0.99
    assert scalars["factor_1/r2"] > 0.99

    # Orthogonality should still be near-zero (computed from coefficients only, not intercepts)
    threshold = _compute_orthogonality_threshold(x, factor_0, factor_1)

    assert "orthogonality_0_1/subspace_overlap" in scalars
    assert "orthogonality_0_1/max_singular_value" in scalars

    overlap = scalars["orthogonality_0_1/subspace_overlap"]
    assert overlap < threshold, f"subspace_overlap={overlap} >= threshold={threshold}"

    max_sv = scalars["orthogonality_0_1/max_singular_value"]
    assert max_sv < threshold, f"max_singular_value={max_sv} >= threshold={threshold}"

    # The different intercepts should not affect orthogonality
    svs = arrays["orthogonality_0_1/singular_values"]
    assert jnp.all(svs < threshold), f"singular_values={svs} not all < threshold={threshold}"


def test_linear_regression_constant_targets_r2_and_dist() -> None:
    """Constant targets should yield r2==0, and dist matches weighted residual norm.

    With intercept: perfect fit to constant -> zero residuals but r2 fallback to 0.0.
    Without intercept: nonzero residuals; verify `dist` against manual computation.
    """
    x = jnp.arange(4.0)[:, None]
    y = jnp.ones_like(x) * 3.0
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])

    # With intercept -> perfect constant fit, but r2 should fallback to 0.0 when variance is zero
    scalars, _ = linear_regression(x, y, weights)
    assert scalars["r2"] == 0.0
    assert jnp.isclose(scalars["rmse"], 0.0, atol=1e-6, rtol=0.0).item()
    assert jnp.isclose(scalars["mae"], 0.0, atol=1e-6, rtol=0.0).item()
    assert jnp.isclose(scalars["dist"], 0.0, atol=1e-6, rtol=0.0).item()

    # Without intercept -> cannot fit a constant perfectly; r2 still 0.0, and dist should match manual computation
    scalars_no_int, arrays_no_int = linear_regression(x, y, weights, fit_intercept=False)
    assert scalars_no_int["r2"] == 0.0
    residuals = arrays_no_int["projected"] - y
    per_sample = jnp.sqrt(jnp.sum(residuals**2, axis=1))
    expected_dist = float(jnp.sum(per_sample * weights))
    assert jnp.isclose(scalars_no_int["dist"], expected_dist, atol=1e-6, rtol=0.0).item()


def test_linear_regression_intercept_and_shapes_both_solvers() -> None:
    """Validate intercept presence/absence and array shapes for both solvers."""
    n, d, t = 5, 3, 2
    x = jnp.arange(float(n * d)).reshape(n, d)
    # Construct multi-target y with known linear relation and intercept
    true_coeffs = jnp.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0]])  # (d, t)
    true_intercept = jnp.array([[0.7, -0.3]])  # (1, t)
    y = x @ true_coeffs + true_intercept
    weights = jnp.ones(n) / n

    # Standard solver, with intercept
    _, arrays = linear_regression(x, y, weights, fit_intercept=True)
    assert "projected" in arrays
    assert "coeffs" in arrays
    assert "intercept" in arrays
    assert arrays["projected"].shape == (n, t)
    assert arrays["coeffs"].shape == (d, t)
    assert arrays["intercept"].shape == (1, t)

    # Standard solver, without intercept
    _, arrays_no_int = linear_regression(x, y, weights, fit_intercept=False)
    assert "projected" in arrays_no_int
    assert "coeffs" in arrays_no_int
    assert "intercept" not in arrays_no_int
    assert arrays_no_int["projected"].shape == (n, t)
    assert arrays_no_int["coeffs"].shape == (d, t)

    # SVD solver, with intercept
    _, arrays_svd = linear_regression_svd(x, y, weights, fit_intercept=True)
    assert "projected" in arrays_svd
    assert "coeffs" in arrays_svd
    assert "intercept" in arrays_svd
    assert arrays_svd["projected"].shape == (n, t)
    assert arrays_svd["coeffs"].shape == (d, t)
    assert arrays_svd["intercept"].shape == (1, t)

    # SVD solver, without intercept
    _, arrays_svd_no_int = linear_regression_svd(x, y, weights, fit_intercept=False)
    assert "projected" in arrays_svd_no_int
    assert "coeffs" in arrays_svd_no_int
    assert "intercept" not in arrays_svd_no_int
    assert arrays_svd_no_int["projected"].shape == (n, t)
    assert arrays_svd_no_int["coeffs"].shape == (d, t)


def test_layer_linear_regression_concat_vs_separate_equivalence() -> None:
    """Concat and separate factor regressions should yield identical per-factor arrays."""
    n, d = 6, 3
    x = jnp.arange(float(n * d)).reshape(n, d)
    # Two factors with different output dims
    w_0 = jnp.array([[1.0, 0.5], [0.0, -1.0], [2.0, 1.0]])  # (d, 2)
    b0 = jnp.array([[0.3, -0.2]])  # (1, 2)
    factor_0 = x @ w_0 + b0

    w_1 = jnp.array([[0.2, 0.0, -0.5], [1.0, 1.0, 0.0], [-1.0, 0.5, 0.3]])  # (d, 3)
    b1 = jnp.array([[0.1, 0.2, -0.1]])  # (1, 3)
    factor_1 = x @ w_1 + b1

    factored_beliefs = (factor_0, factor_1)
    weights = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])

    # Separate per-factor regression
    _, arrays_sep = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        concat_belief_states=False,
    )

    # Concatenated regression with splitting
    _, arrays_cat = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        concat_belief_states=True,
    )

    # Concat path should also provide combined arrays
    assert "concat/projected" in arrays_cat
    assert "concat/coeffs" in arrays_cat
    assert "concat/intercept" in arrays_cat

    # Per-factor arrays should match between separate and concatenated flows
    for k in ["projected", "coeffs", "intercept"]:
        chex.assert_trees_all_close(arrays_sep[f"factor_0/{k}"], arrays_cat[f"factor_0/{k}"])
        chex.assert_trees_all_close(arrays_sep[f"factor_1/{k}"], arrays_cat[f"factor_1/{k}"])


def test_layer_linear_regression_svd_concat_vs_separate_equivalence_best_rcond() -> None:
    """SVD regression: concat-split vs separate produce identical per-factor arrays.

    If belief concatenation is enabled, we only report rcond for the concatenated fit as "concat/best_rcond".
    If belief concatenation is disabled, we report rcond for each factor as "factor_k/best_rcond".
    """
    n, d = 6, 3
    x = jnp.arange(float(n * d)).reshape(n, d)
    # Two factors with different output dims
    w_0 = jnp.array([[1.0, 0.5], [0.0, -1.0], [2.0, 1.0]])  # (d, 2)
    b0 = jnp.array([[0.3, -0.2]])  # (1, 2)
    factor_0 = x @ w_0 + b0

    w_1 = jnp.array([[0.2, 0.0, -0.5], [1.0, 1.0, 0.0], [-1.0, 0.5, 0.3]])  # (d, 3)
    b1 = jnp.array([[0.1, 0.2, -0.1]])  # (1, 3)
    factor_1 = x @ w_1 + b1

    factored_beliefs = (factor_0, factor_1)
    weights = jnp.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])

    # Separate per-factor SVD regression
    scalars_sep, arrays_sep = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        concat_belief_states=False,
        use_svd=True,
        rcond_values=[1e-3],
    )

    # Concatenated SVD regression with splitting
    scalars_cat, arrays_cat = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        concat_belief_states=True,
        use_svd=True,
        rcond_values=[1e-3],
    )

    # Concat path should provide combined arrays and best_rcond
    assert "concat/projected" in arrays_cat
    assert "concat/coeffs" in arrays_cat
    assert "concat/intercept" in arrays_cat
    assert "concat/best_rcond" in scalars_cat
    assert scalars_cat["concat/best_rcond"] == pytest.approx(1e-3)

    # Separate path should include per-factor best_rcond; concat-split path should not
    assert "factor_0/best_rcond" in scalars_sep
    assert "factor_1/best_rcond" in scalars_sep
    assert "factor_0/best_rcond" not in scalars_cat
    assert "factor_1/best_rcond" not in scalars_cat

    # Per-factor arrays should match between separate and concat-split flows
    for k in ["projected", "coeffs", "intercept"]:
        chex.assert_trees_all_close(arrays_sep[f"factor_0/{k}"], arrays_cat[f"factor_0/{k}"])
        chex.assert_trees_all_close(arrays_sep[f"factor_1/{k}"], arrays_cat[f"factor_1/{k}"])

    # Overlapping scalar metrics should agree closely across flows
    for metric in ["r2", "rmse", "mae", "dist"]:
        assert jnp.isclose(
            jnp.asarray(scalars_sep[f"factor_0/{metric}"]),
            jnp.asarray(scalars_cat[f"factor_0/{metric}"]),
            atol=1e-6,
            rtol=0.0,
        ).item()
        assert jnp.isclose(
            jnp.asarray(scalars_sep[f"factor_1/{metric}"]),
            jnp.asarray(scalars_cat[f"factor_1/{metric}"]),
            atol=1e-6,
            rtol=0.0,
        ).item()
