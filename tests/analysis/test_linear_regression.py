"""Tests for reusable linear regression helpers."""

import chex
import jax.numpy as jnp
import pytest

from simplexity.analysis.linear_regression import (
    layer_linear_regression,
    layer_linear_regression_svd,
    linear_regression,
    linear_regression_svd,
)


def test_linear_regression_perfect_fit() -> None:
    """Verify weighted least squares recovers a perfect linear relation."""
    x = jnp.arange(6.0).reshape(-1, 1)
    y = 3.0 * x + 2.0
    weights = jnp.ones(x.shape[0])

    scalars, projections = linear_regression(x, y, weights)

    assert pytest.approx(1.0) == scalars["r2"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["rmse"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["mae"]
    chex.assert_trees_all_close(projections["projected"], y)


def test_linear_regression_svd_selects_best_rcond() -> None:
    """Ensure the SVD variant exposes chosen rcond and predictions."""
    x = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]])
    y = jnp.sum(x, axis=1, keepdims=True)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])

    scalars, projections = linear_regression_svd(
        x,
        y,
        weights,
        rcond_values=[1e-6, 1e-4, 1e-2],
    )

    assert scalars["best_rcond"] in {1e-6, 1e-4, 1e-2}
    chex.assert_trees_all_close(projections["projected"], y)


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

    scalars, projections = linear_regression(x, y, None, fit_intercept=False)

    assert pytest.approx(1.0) == scalars["r2"]
    chex.assert_trees_all_close(projections["projected"], y)


def test_linear_regression_svd_handles_empty_features() -> None:
    """SVD helper should handle inputs with no feature columns."""
    x = jnp.empty((3, 0))
    y = jnp.arange(3.0)[:, None]
    weights = jnp.ones(3)

    scalars, projections = linear_regression_svd(x, y, weights, fit_intercept=False)

    assert scalars["best_rcond"] == pytest.approx(1e-15)
    chex.assert_trees_all_close(projections["projected"], jnp.zeros_like(y))


def test_linear_regression_accepts_one_dimensional_inputs() -> None:
    """1D features and targets should be promoted to column vectors."""
    x = jnp.arange(4.0)
    y = 5.0 * x + 1.0
    weights = jnp.ones_like(x)

    scalars, projections = linear_regression(x, y, weights)

    assert pytest.approx(1.0) == scalars["r2"]
    chex.assert_trees_all_close(projections["projected"], y[:, None])


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

    scalars, projections = layer_linear_regression_svd(
        x,
        weights,
        beliefs,
        rcond_values=[1e-3],
    )

    assert pytest.approx(1.0, abs=1e-6) == scalars["r2"]
    chex.assert_trees_all_close(projections["projected"], beliefs)


def test_layer_linear_regression_to_factors_basic() -> None:
    """Layer regression with to_factors should regress to each factor separately."""
    x = jnp.arange(12.0).reshape(4, 3)  # 4 samples, 3 features
    weights = jnp.ones(4) / 4.0

    # Two factors: factor 0 has 2 states, factor 1 has 3 states
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])  # [4, 2]
    factor_1 = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])  # [4, 3]
    factored_beliefs = (factor_0, factor_1)

    scalars, projections = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        to_factors=True,
    )

    # Should have separate metrics for each factor
    assert "r2_factor_0" in scalars
    assert "r2_factor_1" in scalars
    assert "rmse_factor_0" in scalars
    assert "rmse_factor_1" in scalars
    assert "mae_factor_0" in scalars
    assert "mae_factor_1" in scalars
    assert "dist_factor_0" in scalars
    assert "dist_factor_1" in scalars

    # Should have separate projections for each factor
    assert "projected_factor_0" in projections
    assert "projected_factor_1" in projections

    # Check shapes
    assert projections["projected_factor_0"].shape == factor_0.shape
    assert projections["projected_factor_1"].shape == factor_1.shape


def test_layer_linear_regression_svd_to_factors_basic() -> None:
    """Layer regression SVD with to_factors should regress to each factor separately."""
    x = jnp.arange(12.0).reshape(4, 3)  # 4 samples, 3 features
    weights = jnp.ones(4) / 4.0

    # Two factors: factor 0 has 2 states, factor 1 has 3 states
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])  # [4, 2]
    factor_1 = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])  # [4, 3]
    factored_beliefs = (factor_0, factor_1)

    scalars, projections = layer_linear_regression_svd(
        x,
        weights,
        factored_beliefs,
        to_factors=True,
        rcond_values=[1e-6],
    )

    # Should have separate metrics for each factor including best_rcond
    assert "r2_factor_0" in scalars
    assert "r2_factor_1" in scalars
    assert "best_rcond_factor_0" in scalars
    assert "best_rcond_factor_1" in scalars

    # Should have separate projections for each factor
    assert "projected_factor_0" in projections
    assert "projected_factor_1" in projections

    # Check shapes
    assert projections["projected_factor_0"].shape == factor_0.shape
    assert projections["projected_factor_1"].shape == factor_1.shape


def test_layer_linear_regression_to_factors_single_factor() -> None:
    """to_factors=True should work with a single factor tuple."""
    x = jnp.arange(9.0).reshape(3, 3)
    weights = jnp.ones(3) / 3.0

    # Single factor in tuple
    factor_0 = jnp.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
    factored_beliefs = (factor_0,)

    scalars, projections = layer_linear_regression(
        x,
        weights,
        factored_beliefs,
        to_factors=True,
    )

    # Should have metrics for single factor
    assert "r2_factor_0" in scalars
    assert "projected_factor_0" in projections
    assert projections["projected_factor_0"].shape == factor_0.shape


def test_layer_linear_regression_to_factors_requires_tuple() -> None:
    """to_factors=True requires belief_states to be a tuple."""
    x = jnp.ones((3, 2))
    weights = jnp.ones(3) / 3.0
    beliefs_array = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="belief_states must be a tuple when to_factors is True"):
        layer_linear_regression(x, weights, beliefs_array, to_factors=True)

    with pytest.raises(ValueError, match="belief_states must be a tuple when to_factors is True"):
        layer_linear_regression_svd(x, weights, beliefs_array, to_factors=True)


def test_layer_linear_regression_to_factors_validates_tuple_contents() -> None:
    """to_factors=True requires all elements in tuple to be jax.Arrays."""
    x = jnp.ones((3, 2))
    weights = jnp.ones(3) / 3.0

    # Invalid: tuple contains non-array
    invalid_beliefs = (jnp.ones((3, 2)), "not an array")

    with pytest.raises(ValueError, match="Each factor in belief_states must be a jax.Array"):
        layer_linear_regression(x, weights, invalid_beliefs, to_factors=True)

    with pytest.raises(ValueError, match="Each factor in belief_states must be a jax.Array"):
        layer_linear_regression_svd(x, weights, invalid_beliefs, to_factors=True)


def test_layer_linear_regression_to_factors_false_requires_array() -> None:
    """to_factors=False requires belief_states to be a single array, not a tuple."""
    x = jnp.ones((3, 2))
    weights = jnp.ones(3) / 3.0

    # Invalid: tuple when to_factors=False
    factored_beliefs = (jnp.ones((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError, match="belief_states must be a single array when to_factors is False"):
        layer_linear_regression(x, weights, factored_beliefs, to_factors=False)

    with pytest.raises(ValueError, match="belief_states must be a single array when to_factors is False"):
        layer_linear_regression_svd(x, weights, factored_beliefs, to_factors=False)
