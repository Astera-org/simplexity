"""Tests for PCA analysis."""

import jax.numpy as jnp
import numpy as np
import pytest
import plotly.graph_objects as go

from simplexity.analysis.pca import (
    compute_pca,
    compute_variance_thresholds,
    plot_pca_2d,
    plot_pca_3d,
    plot_cumulative_explained_variance,
    generate_pca_plots,
    plot_pca_2d_with_step_slider,
    plot_pca_2d_with_layer_dropdown,
    plot_pca_2d_with_step_and_layer,
    plot_cumulative_variance_with_step_dropdown,
    plot_cumulative_variance_with_layer_dropdown,
)


class TestComputePCA:
    def test_basic_pca(self):
        """Test basic PCA computation."""
        # Simple 2D data
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])

        result = compute_pca(X, n_components=2)

        # Check result structure
        assert "components" in result
        assert "explained_variance" in result
        assert "explained_variance_ratio" in result
        assert "mean" in result
        assert "X_proj" in result

        # Check shapes
        assert result["components"].shape == (2, 2)  # (n_components, n_features)
        assert result["explained_variance"].shape == (2,)
        assert result["X_proj"].shape == (4, 2)  # (n_samples, n_components)

    def test_pca_with_weights(self):
        """Test weighted PCA."""
        X = np.random.randn(10, 5)
        weights = np.random.rand(10)

        result = compute_pca(X, n_components=3, weights=weights)

        assert result["components"].shape == (3, 5)
        assert result["X_proj"].shape == (10, 3)

    def test_explained_variance_sums_to_one(self):
        """Test that explained variance ratios sum to ~1."""
        X = np.random.randn(20, 10)
        result = compute_pca(X, n_components=10)

        total_variance = result["explained_variance_ratio"].sum()
        assert np.isclose(total_variance, 1.0, atol=1e-6)

    def test_pca_without_centering(self):
        """Test PCA without mean centering."""
        X = np.random.randn(10, 5)
        result = compute_pca(X, center=False)

        # Mean should be zero
        assert np.allclose(result["mean"], 0.0)

    def test_pca_n_components_auto(self):
        """Test PCA with automatic component selection."""
        X = np.random.randn(20, 10)
        result = compute_pca(X)  # n_components=None

        # Should use all components
        assert result["components"].shape[0] == 10

    def test_pca_n_components_exceeds_features(self):
        """Test PCA when n_components > n_features."""
        X = np.random.randn(20, 5)
        result = compute_pca(X, n_components=10)

        # Should cap at n_features
        assert result["components"].shape[0] == 5

    def test_pca_with_jax_array(self):
        """Test PCA with JAX array input."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        result = compute_pca(X, n_components=2)

        assert result["components"].shape == (2, 2)


class TestComputeVarianceThresholds:
    def test_variance_thresholds(self):
        """Test computing variance thresholds."""
        # Create dummy PCA result
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=10)

        thresholds = [0.5, 0.8, 0.9, 0.95]
        result = compute_variance_thresholds(pca_res, thresholds)

        # Check all thresholds are present
        assert len(result) == len(thresholds)
        for threshold in thresholds:
            assert threshold in result
            assert isinstance(result[threshold], int)
            assert result[threshold] > 0

    def test_variance_thresholds_ordering(self):
        """Test that higher thresholds require more components."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=10)

        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        result = compute_variance_thresholds(pca_res, thresholds)

        # Higher thresholds should require >= components
        assert result[0.5] <= result[0.8]
        assert result[0.8] <= result[0.9]
        assert result[0.9] <= result[0.95]
        assert result[0.95] <= result[0.99]

    def test_unreachable_threshold(self):
        """Test threshold that cannot be reached."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=5)

        # If max explained variance is < 1.0, threshold of 1.0 is unreachable
        result = compute_variance_thresholds(pca_res, [1.0])

        # Should return max components from all_explained_variance_ratio
        # which has shape (D,) not (n_components,)
        assert result[1.0] == len(pca_res["all_explained_variance_ratio"])


class TestPlotPCA2D:
    def test_basic_2d_plot(self):
        """Test basic 2D PCA plot generation."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=2)

        fig = plot_pca_2d(pca_res)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_2d_plot_with_labels(self):
        """Test 2D plot with labels."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=2)
        labels = np.random.randint(0, 3, size=50)

        fig = plot_pca_2d(pca_res, labels=labels)

        assert isinstance(fig, go.Figure)


class TestPlotPCA3D:
    def test_basic_3d_plot(self):
        """Test basic 3D PCA plot generation."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=3)

        fig = plot_pca_3d(pca_res)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotCumulativeExplainedVariance:
    def test_cumulative_variance_plot(self):
        """Test cumulative explained variance plot."""
        X = np.random.randn(50, 10)
        pca_res = compute_pca(X, n_components=10)

        fig = plot_cumulative_explained_variance(pca_res)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestGeneratePCAPlots:
    def test_generate_all_plots(self):
        """Test generating all PCA plots."""
        X = np.random.randn(50, 10)

        plots = generate_pca_plots(
            X,
            n_components=3,
            weights=None,
            plot_2d=True,
            plot_3d=True,
            plot_cumulative_variance=True,
        )

        assert "pca_2d" in plots
        assert "pca_3d" in plots
        assert "cumulative_explained_variance" in plots

    def test_generate_selective_plots(self):
        """Test generating only selected plots."""
        X = np.random.randn(50, 10)

        plots = generate_pca_plots(
            X, plot_2d=True, plot_3d=False, plot_cumulative_variance=False
        )

        assert "pca_2d" in plots
        assert "pca_3d" not in plots
        assert "cumulative_explained_variance" not in plots


class TestPlotPCAWithStepSlider:
    def test_step_slider_plot(self):
        """Test PCA plot with step slider."""
        # Create PCA results for multiple steps
        pca_results_by_step = {}
        for step in [0, 100, 200]:
            X = np.random.randn(50, 10)
            pca_results_by_step[step] = compute_pca(X, n_components=2)

        fig = plot_pca_2d_with_step_slider(pca_results_by_step)

        assert isinstance(fig, go.Figure)
        # Should have one trace per step
        assert len(fig.data) == 3
        # Should have slider
        assert len(fig.layout.sliders) == 1

    def test_step_slider_with_labels(self):
        """Test step slider with labels."""
        pca_results_by_step = {}
        labels_by_step = {}
        for step in [0, 100]:
            X = np.random.randn(50, 10)
            pca_results_by_step[step] = compute_pca(X, n_components=2)
            labels_by_step[step] = np.random.randint(0, 3, size=50)

        fig = plot_pca_2d_with_step_slider(pca_results_by_step, labels_by_step)

        assert isinstance(fig, go.Figure)

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_pca_2d_with_step_slider({})


class TestPlotPCAWithLayerDropdown:
    def test_layer_dropdown_plot(self):
        """Test PCA plot with layer dropdown."""
        pca_results_by_layer = {}
        for layer in ["layer_0", "layer_1", "layer_2"]:
            X = np.random.randn(50, 10)
            pca_results_by_layer[layer] = compute_pca(X, n_components=2)

        fig = plot_pca_2d_with_layer_dropdown(pca_results_by_layer)

        assert isinstance(fig, go.Figure)
        # Should have one trace per layer
        assert len(fig.data) == 3
        # Should have dropdown menu
        assert len(fig.layout.updatemenus) == 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_pca_2d_with_layer_dropdown({})


class TestPlotPCAWithStepAndLayer:
    def test_combined_plot(self):
        """Test PCA plot with both step slider and layer dropdown."""
        pca_results_by_step_and_layer = {}
        for step in [0, 100]:
            pca_results_by_step_and_layer[step] = {}
            for layer in ["layer_0", "layer_1"]:
                X = np.random.randn(50, 10)
                pca_results_by_step_and_layer[step][layer] = compute_pca(
                    X, n_components=2
                )

        fig = plot_pca_2d_with_step_and_layer(pca_results_by_step_and_layer)

        assert isinstance(fig, go.Figure)
        # Should have traces for each (step, layer) combination
        assert len(fig.data) == 4  # 2 steps × 2 layers
        # Should have both slider and dropdown
        assert len(fig.layout.sliders) == 1
        assert len(fig.layout.updatemenus) == 1

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_pca_2d_with_step_and_layer({})


class TestPlotCumulativeVarianceWithStepDropdown:
    def test_step_dropdown_plot(self):
        """Test cumulative variance plot with step dropdown."""
        X = np.random.randn(50, 10)

        # Create results for multiple steps and layers
        pca_results_by_step_and_layer = {}
        for step in [0, 1, 2]:
            pca_results_by_step_and_layer[step] = {}
            for layer in ["layer_0", "layer_1"]:
                pca_results_by_step_and_layer[step][layer] = compute_pca(
                    X, n_components=5
                )

        fig = plot_cumulative_variance_with_step_dropdown(
            pca_results_by_step_and_layer,
            thresholds=[0.8, 0.9],
        )

        assert isinstance(fig, go.Figure)
        # Should have dropdown for steps
        assert len(fig.layout.updatemenus) == 1
        # Should have 3 buttons (one per step)
        assert len(fig.layout.updatemenus[0].buttons) == 3

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_cumulative_variance_with_step_dropdown({})


class TestPlotCumulativeVarianceWithLayerDropdown:
    def test_layer_dropdown_plot(self):
        """Test cumulative variance plot with layer dropdown."""
        X = np.random.randn(50, 10)

        # Create results for multiple steps and layers
        pca_results_by_step_and_layer = {}
        for step in [0, 1, 2]:
            pca_results_by_step_and_layer[step] = {}
            for layer in ["layer_0", "layer_1"]:
                pca_results_by_step_and_layer[step][layer] = compute_pca(
                    X, n_components=5
                )

        fig = plot_cumulative_variance_with_layer_dropdown(
            pca_results_by_step_and_layer,
            thresholds=[0.8, 0.9],
        )

        assert isinstance(fig, go.Figure)
        # Should have dropdown for layers
        assert len(fig.layout.updatemenus) == 1
        # Should have 2 buttons (one per layer)
        assert len(fig.layout.updatemenus[0].buttons) == 2

    def test_empty_results_raises_error(self):
        """Test that empty results raise an error."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_cumulative_variance_with_layer_dropdown({})
