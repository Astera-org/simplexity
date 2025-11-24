"""Tests for reusable linear regression helpers."""

import chex
import numpy as np
import pytest

from simplexity.analysis.linear_regression import (
    layer_linear_regression,
    layer_linear_regression_svd,
    linear_regression,
    linear_regression_svd,
)


def test_linear_regression_perfect_fit() -> None:
    """Verify weighted least squares recovers a perfect linear relation."""
    x = np.arange(6.0).reshape(-1, 1)
    y = 3.0 * x + 2.0
    weights = np.ones(x.shape[0])

    scalars, projections = linear_regression(x, y, weights)

    assert pytest.approx(1.0) == scalars["r2"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["rmse"]
    assert pytest.approx(0.0, abs=1e-5) == scalars["mae"]
    chex.assert_trees_all_close(projections["projected"], y)


def test_linear_regression_svd_selects_best_rcond() -> None:
    """Ensure the SVD variant exposes chosen rcond and predictions."""
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]])
    y = np.sum(x, axis=1, keepdims=True)
    weights = np.array([0.1, 0.2, 0.3, 0.4])

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
    x = np.ones((3, 2))
    weights = np.ones(3)

    with pytest.raises(ValueError, match="requires belief_states"):
        layer_linear_regression(x, weights, None)

    with pytest.raises(ValueError, match="requires belief_states"):
        layer_linear_regression_svd(x, weights, None)


def test_linear_regression_rejects_mismatched_weights() -> None:
    """Weights must align with the sample dimension."""
    x = np.ones((4, 1))
    y = np.ones((4, 1))
    weights = np.ones(3)

    with pytest.raises(ValueError, match="Weights must be shape"):
        linear_regression(x, y, weights)


def test_linear_regression_rejects_negative_weights() -> None:
    """Negative weights should be rejected before fitting."""
    x = np.ones((4, 1))
    y = np.ones((4, 1))
    weights = np.array([0.5, -0.1, 0.3, 0.3])

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        linear_regression(x, y, weights)


def test_linear_regression_rejects_zero_sum_weights() -> None:
    """Weight normalization should fail when the sum is zero."""
    x = np.ones((2, 1))
    y = np.ones((2, 1))
    weights = np.array([0.0, 0.0])

    with pytest.raises(ValueError, match="Weights must sum to a positive value"):
        linear_regression(x, y, weights)


def test_linear_regression_without_intercept_uses_uniform_weights() -> None:
    """When weights are None the helper should apply uniform weighting."""
    x = np.arange(1.0, 4.0)[:, None]
    y = 2.0 * x

    scalars, projections = linear_regression(x, y, None, fit_intercept=False)

    assert pytest.approx(1.0) == scalars["r2"]
    chex.assert_trees_all_close(projections["projected"], y)


def test_linear_regression_svd_handles_empty_features() -> None:
    """SVD helper should handle inputs with no feature columns."""
    x = np.empty((3, 0))
    y = np.arange(3.0)[:, None]
    weights = np.ones(3)

    scalars, projections = linear_regression_svd(x, y, weights, fit_intercept=False)

    assert scalars["best_rcond"] == pytest.approx(1e-15)
    chex.assert_trees_all_close(projections["projected"], np.zeros_like(y))
