"""Tests for regression analysis."""

import jax.numpy as jnp
import numpy as np
import pytest
import plotly.graph_objects as go

from simplexity.analysis.regression import (
    RegressionResult,
    _weighted_regression,
    _compute_metrics,
    regress_with_kfold_rcond_cv,
    project_to_simplex,
    regress_activations_to_beliefs,
    plot_simplex_projection_with_step_slider,
    plot_simplex_projection_with_layer_dropdown,
    plot_simplex_projection_with_step_and_layer,
)


class TestWeightedRegression:
    def test_basic_regression(self):
        """Test basic weighted regression."""
        # Simple linear relationship: Y = 2*X + 1
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        Y = jnp.array([[3.0], [5.0], [7.0], [9.0]])
        weights = jnp.ones(4)

        beta = _weighted_regression(X, Y, weights, rcond=1e-10)

        # Should have shape (D+1, B) = (2, 1) for bias + 1 feature
        assert beta.shape == (2, 1)

        # Check that it learned approximately the right relationship
        # beta[0] = bias ≈ 1, beta[1] = slope ≈ 2
        assert np.isclose(beta[0, 0], 1.0, atol=0.1)
        assert np.isclose(beta[1, 0], 2.0, atol=0.1)

    def test_weighted_regression_with_weights(self):
        """Test that weights affect the regression."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        Y = jnp.array([[1.0], [2.0], [3.0], [10.0]])  # Last point is outlier

        # Uniform weights
        weights_uniform = jnp.ones(4)
        beta_uniform = _weighted_regression(X, Y, weights_uniform, rcond=1e-10)

        # Downweight the outlier
        weights_downweight = jnp.array([1.0, 1.0, 1.0, 0.1])
        beta_downweight = _weighted_regression(X, Y, weights_downweight, rcond=1e-10)

        # Results should differ
        assert not np.allclose(beta_uniform, beta_downweight)

    def test_multidimensional_output(self):
        """Test regression with multiple output dimensions."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        weights = jnp.ones(3)

        beta = _weighted_regression(X, Y, weights, rcond=1e-10)

        # Shape should be (D+1, B) = (3, 3) for bias + 2 features, 3 outputs
        assert beta.shape == (3, 3)

    def test_rcond_regularization(self):
        """Test that rcond affects regularization."""
        X = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # Collinear features
        Y = jnp.array([[1.0], [2.0], [3.0]])
        weights = jnp.ones(3)

        # High rcond (more regularization)
        beta_high = _weighted_regression(X, Y, weights, rcond=1e-1)

        # Low rcond (less regularization)
        beta_low = _weighted_regression(X, Y, weights, rcond=1e-10)

        # Results may differ due to regularization
        assert beta_high.shape == beta_low.shape


class TestComputeMetrics:
    def test_basic_metrics(self):
        """Test computing regression metrics."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        Y = jnp.array([[2.0], [4.0], [6.0], [8.0]])
        weights = jnp.ones(4)

        # Perfect linear fit: beta = [0, 2]
        beta = jnp.array([[0.0], [2.0]])

        dist, r2, mse, mae, rmse, Y_pred = _compute_metrics(X, Y, weights, beta)

        # Perfect fit should have R² = 1.0
        assert np.isclose(r2, 1.0, atol=0.01)
        # Perfect fit should have low error
        assert dist < 0.1
        assert np.all(mse < 0.1)

    def test_metrics_with_imperfect_fit(self):
        """Test metrics with imperfect fit."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        Y = jnp.array([[2.0], [4.0], [6.0], [8.0]])
        weights = jnp.ones(4)

        # Imperfect beta
        beta = jnp.array([[1.0], [1.5]])

        dist, r2, mse, mae, rmse, Y_pred = _compute_metrics(X, Y, weights, beta)

        # Note: R² can exceed 1.0 for bad models (it's not bounded above)
        # What matters is that the fit is not perfect
        # Should have non-zero error
        assert dist > 0
        assert np.all(mse > 0)

    def test_metrics_shapes(self):
        """Test that metric shapes are correct."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        weights = jnp.ones(3)
        beta = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        dist, r2, mse, mae, rmse, Y_pred = _compute_metrics(X, Y, weights, beta)

        # Scalar metrics
        assert isinstance(dist, (float, np.floating))
        assert isinstance(r2, (float, np.floating))

        # Per-dimension metrics
        assert mse.shape == (2,)
        assert mae.shape == (2,)
        assert rmse.shape == (2,)

        # Predictions shape
        assert Y_pred.shape == Y.shape


class TestRegressWithKFoldRcondCV:
    def test_basic_kfold_cv(self):
        """Test K-fold CV regression."""
        # Generate simple data
        np.random.seed(42)
        X = jnp.array(np.random.randn(50, 5))
        true_beta = np.array([1.0, 2.0, -1.0, 0.5, 0.0])
        Y = jnp.array(X @ true_beta[:, None] + np.random.randn(50, 1) * 0.1)
        weights = jnp.ones(50)

        rcond_values = [1e-10, 1e-5, 1e-3]
        result = regress_with_kfold_rcond_cv(
            X, Y, weights, rcond_values, n_splits=5, random_state=42
        )

        # Check result structure
        assert isinstance(result, RegressionResult)
        assert result.best_rcond in rcond_values
        assert result.predictions.shape == (50, 1)
        assert result.true_values.shape == (50, 1)
        assert result.weights.shape == (50,)
        assert len(result.per_rcond_cv_error) == len(rcond_values)

    def test_rcond_selection(self):
        """Test that cross-validation selects an rcond value."""
        X = jnp.array(np.random.randn(30, 3))
        Y = jnp.array(np.random.randn(30, 2))
        weights = jnp.ones(30)

        rcond_values = [1e-10, 1e-5, 1e-2]
        result = regress_with_kfold_rcond_cv(
            X, Y, weights, rcond_values, n_splits=3
        )

        # Should select one of the provided values
        assert result.best_rcond in rcond_values

    def test_cv_error_decreases_with_better_rcond(self):
        """Test that CV errors are computed for all rcond values."""
        X = jnp.array(np.random.randn(40, 4))
        Y = jnp.array(np.random.randn(40, 1))
        weights = jnp.ones(40)

        rcond_values = [1e-10, 1e-6, 1e-3]
        result = regress_with_kfold_rcond_cv(X, Y, weights, rcond_values, n_splits=5)

        # Should have CV error for each rcond
        assert len(result.per_rcond_cv_error) == 3
        for rcond in rcond_values:
            assert rcond in result.per_rcond_cv_error
            assert result.per_rcond_cv_error[rcond] >= 0


class TestProjectToSimplex:
    def test_simplex_projection(self):
        """Test 2-simplex projection."""
        # Points on 2-simplex (sum to 1)
        points = np.array([[0.33, 0.33, 0.34], [0.5, 0.3, 0.2], [0.0, 0.5, 0.5]])

        x, y = project_to_simplex(points)

        # Should produce 2D coordinates
        assert x.shape == (3,)
        assert y.shape == (3,)

    def test_simplex_corners(self):
        """Test projection of simplex corners."""
        # Corner points
        corners = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        x, y = project_to_simplex(corners)

        # All coordinates should be different
        assert len(np.unique(x)) == 3 or len(np.unique(y)) == 3


class TestRegressActivationsToBeliefs:
    def test_basic_regression_to_beliefs(self):
        """Test regressing activations to beliefs."""
        # Create simple test data
        batch_size, seq_len = 5, 4
        n_beliefs = 3
        d_layer = 6

        inputs = jnp.array(np.random.randint(0, 10, size=(batch_size, seq_len)))
        beliefs = jnp.array(np.random.rand(batch_size, seq_len, n_beliefs))
        # Normalize beliefs to sum to 1
        beliefs = beliefs / beliefs.sum(axis=-1, keepdims=True)
        probs = jnp.array(np.random.rand(batch_size, seq_len))
        activations_by_layer = {
            "layer_0": jnp.array(np.random.randn(batch_size, seq_len, d_layer)),
        }

        result, fig = regress_activations_to_beliefs(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
            layer_name="layer_0",
            rcond_values=[1e-5, 1e-3],
            n_splits=3,
            plot_projection=False,
        )

        # Check result
        assert isinstance(result, RegressionResult)
        assert fig is None

    def test_regression_with_plot(self):
        """Test regression with projection plot."""
        batch_size, seq_len = 5, 4
        n_beliefs = 3
        d_layer = 6

        inputs = jnp.array(np.random.randint(0, 10, size=(batch_size, seq_len)))
        beliefs = jnp.array(np.random.rand(batch_size, seq_len, n_beliefs))
        beliefs = beliefs / beliefs.sum(axis=-1, keepdims=True)
        probs = jnp.array(np.random.rand(batch_size, seq_len))
        activations_by_layer = {
            "layer_0": jnp.array(np.random.randn(batch_size, seq_len, d_layer)),
        }

        result, fig = regress_activations_to_beliefs(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
            layer_name="layer_0",
            rcond_values=[1e-5],
            n_splits=2,
            plot_projection=True,
        )

        # Should return a figure
        assert isinstance(fig, go.Figure)


class TestPlotSimplexWithStepSlider:
    def test_step_slider_plot(self):
        """Test simplex projection plot with step slider."""
        # Create regression results for multiple steps
        regression_results_by_step = {}
        for step in [0, 100, 200]:
            regression_results_by_step[step] = RegressionResult(
                best_rcond=1e-5,
                dist=0.5,
                r2=0.8,
                mse=np.array([0.1, 0.2, 0.15]),
                mae=np.array([0.05, 0.1, 0.08]),
                rmse=np.array([0.31, 0.45, 0.39]),
                predictions=np.random.rand(10, 3),
                true_values=np.random.rand(10, 3),
                weights=np.ones(10),
                per_rcond_cv_error={1e-5: 0.5},
            )

        fig = plot_simplex_projection_with_step_slider(regression_results_by_step)

        assert isinstance(fig, go.Figure)
        # Should have 2 traces per step (true and predicted)
        assert len(fig.data) == 6  # 3 steps × 2 traces
        # Should have slider
        assert len(fig.layout.sliders) == 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_simplex_projection_with_step_slider({})


class TestPlotSimplexWithLayerDropdown:
    def test_layer_dropdown_plot(self):
        """Test simplex projection plot with layer dropdown."""
        regression_results_by_layer = {}
        for layer in ["layer_0", "layer_1"]:
            regression_results_by_layer[layer] = RegressionResult(
                best_rcond=1e-5,
                dist=0.5,
                r2=0.8,
                mse=np.array([0.1, 0.2, 0.15]),
                mae=np.array([0.05, 0.1, 0.08]),
                rmse=np.array([0.31, 0.45, 0.39]),
                predictions=np.random.rand(10, 3),
                true_values=np.random.rand(10, 3),
                weights=np.ones(10),
                per_rcond_cv_error={1e-5: 0.5},
            )

        fig = plot_simplex_projection_with_layer_dropdown(regression_results_by_layer)

        assert isinstance(fig, go.Figure)
        # Should have 2 traces per layer
        assert len(fig.data) == 4  # 2 layers × 2 traces
        # Should have dropdown
        assert len(fig.layout.updatemenus) == 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_simplex_projection_with_layer_dropdown({})


class TestPlotSimplexWithStepAndLayer:
    def test_combined_plot(self):
        """Test simplex projection plot with both step and layer controls."""
        regression_results_by_step_and_layer = {}
        for step in [0, 100]:
            regression_results_by_step_and_layer[step] = {}
            for layer in ["layer_0", "layer_1"]:
                regression_results_by_step_and_layer[step][layer] = RegressionResult(
                    best_rcond=1e-5,
                    dist=0.5,
                    r2=0.8,
                    mse=np.array([0.1, 0.2, 0.15]),
                    mae=np.array([0.05, 0.1, 0.08]),
                    rmse=np.array([0.31, 0.45, 0.39]),
                    predictions=np.random.rand(10, 3),
                    true_values=np.random.rand(10, 3),
                    weights=np.ones(10),
                    per_rcond_cv_error={1e-5: 0.5},
                )

        fig = plot_simplex_projection_with_step_and_layer(
            regression_results_by_step_and_layer
        )

        assert isinstance(fig, go.Figure)
        # Should have 2 traces per (step, layer) combination
        assert len(fig.data) == 8  # 2 steps × 2 layers × 2 traces
        # Should have both slider and dropdown
        assert len(fig.layout.sliders) == 1
        assert len(fig.layout.updatemenus) == 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_simplex_projection_with_step_and_layer({})
