"""Tests for analysis tracker."""

import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from simplexity.analysis.tracker import AnalysisTracker
from simplexity.analysis.regression import RegressionResult


class TestAnalysisTracker:
    def setup_method(self):
        """Set up test fixtures."""
        self.layer_names = ["layer_0", "layer_1"]
        self.tracker = AnalysisTracker(
            layer_names=self.layer_names, variance_thresholds=[0.8, 0.9, 0.95]
        )

    def create_dummy_data(self, batch_size=5, seq_len=4):
        """Create dummy data for testing."""
        n_beliefs = 3
        d_layer = 6

        inputs = jnp.array(np.random.randint(0, 10, size=(batch_size, seq_len)))
        beliefs = jnp.array(np.random.rand(batch_size, seq_len, n_beliefs))
        beliefs = beliefs / beliefs.sum(axis=-1, keepdims=True)
        probs = jnp.array(np.random.rand(batch_size, seq_len))
        activations_by_layer = {
            name: jnp.array(np.random.randn(batch_size, seq_len, d_layer))
            for name in self.layer_names
        }

        return inputs, beliefs, probs, activations_by_layer

    def test_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.layer_names == self.layer_names
        assert self.tracker.variance_thresholds == [0.8, 0.9, 0.95]
        assert len(self.tracker.get_steps()) == 0

    def test_add_step(self):
        """Test adding a step to the tracker."""
        inputs, beliefs, probs, activations = self.create_dummy_data()

        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Should have one step
        assert len(self.tracker.get_steps()) == 1
        assert 0 in self.tracker.get_steps()

    def test_add_multiple_steps(self):
        """Test adding multiple steps."""
        for step in [0, 100, 200]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        steps = self.tracker.get_steps()
        assert len(steps) == 3
        assert steps == [0, 100, 200]  # Should be sorted

    def test_get_pca_result(self):
        """Test retrieving PCA results."""
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Should be able to get PCA result
        pca_res = self.tracker.get_pca_result(step=0, layer_name="layer_0")
        assert pca_res is not None
        assert "components" in pca_res
        assert "explained_variance" in pca_res

    def test_get_regression_result(self):
        """Test retrieving regression results."""
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Should be able to get regression result
        reg_res = self.tracker.get_regression_result(step=0, layer_name="layer_0")
        assert reg_res is not None
        assert isinstance(reg_res, RegressionResult)

    def test_get_variance_thresholds(self):
        """Test retrieving variance threshold results."""
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        thresholds = self.tracker.get_variance_thresholds(step=0, layer_name="layer_0")
        assert thresholds is not None
        assert 0.8 in thresholds
        assert 0.9 in thresholds
        assert 0.95 in thresholds

    def test_selective_computation(self):
        """Test computing only selected analyses."""
        inputs, beliefs, probs, activations = self.create_dummy_data()

        # Only PCA
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
            compute_pca=True,
            compute_regression=False,
        )

        pca_res = self.tracker.get_pca_result(step=0, layer_name="layer_0")
        reg_res = self.tracker.get_regression_result(step=0, layer_name="layer_0")

        assert pca_res is not None
        assert reg_res is None

    def test_generate_pca_plots(self):
        """Test generating PCA plots."""
        # Add multiple steps
        for step in [0, 100]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        plots = self.tracker.generate_pca_plots()

        # Should generate plots
        assert len(plots) > 0
        # Combined plot should be present
        assert "pca_combined" in plots

    def test_generate_regression_plots(self):
        """Test generating regression plots."""
        # Add multiple steps
        for step in [0, 100]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        plots = self.tracker.generate_regression_plots()

        # Should generate plots
        assert len(plots) > 0
        # Combined plot should be present
        assert "regression_combined" in plots

    def test_generate_selective_plots(self):
        """Test generating only selected plot types."""
        for step in [0, 100]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        # Only combined plots
        plots = self.tracker.generate_pca_plots(
            by_step=False, by_layer=False, combined=True
        )

        assert "pca_combined" in plots
        # Should not have step-specific plots
        assert not any("step_slider" in key for key in plots.keys())

    def test_get_variance_threshold_summary(self):
        """Test getting variance threshold summary."""
        for step in [0, 100]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        summary = self.tracker.get_variance_threshold_summary()

        # Check structure
        assert "by_layer" in summary
        assert "by_step" in summary

        # Check by_layer
        for layer_name in self.layer_names:
            assert layer_name in summary["by_layer"]
            for threshold in [0.8, 0.9, 0.95]:
                assert threshold in summary["by_layer"][layer_name]
                # Should have values for each step
                assert len(summary["by_layer"][layer_name][threshold]) == 2

        # Check by_step
        for step in [0, 100]:
            assert step in summary["by_step"]
            for layer_name in self.layer_names:
                assert layer_name in summary["by_step"][step]

    def test_get_regression_metrics_summary(self):
        """Test getting regression metrics summary."""
        for step in [0, 100]:
            inputs, beliefs, probs, activations = self.create_dummy_data()
            self.tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations,
            )

        summary = self.tracker.get_regression_metrics_summary()

        # Check structure
        assert "by_layer" in summary
        assert "by_step" in summary

        # Check by_layer
        for layer_name in self.layer_names:
            assert layer_name in summary["by_layer"]
            assert "r2" in summary["by_layer"][layer_name]
            assert "dist" in summary["by_layer"][layer_name]
            assert "best_rcond" in summary["by_layer"][layer_name]
            # Should have values for each step
            assert len(summary["by_layer"][layer_name]["r2"]) == 2

        # Check by_step
        for step in [0, 100]:
            assert step in summary["by_step"]
            for layer_name in self.layer_names:
                assert layer_name in summary["by_step"][step]
                assert "r2" in summary["by_step"][step][layer_name]
                assert "dist" in summary["by_step"][step][layer_name]

    def test_save_all_plots(self):
        """Test saving plots to disk."""
        # Add data
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tracker.save_all_plots(tmpdir)

            # Check that files were created
            files = os.listdir(tmpdir)
            assert len(files) > 0
            # Should have HTML files
            assert any(f.endswith(".html") for f in files)

    def test_clear(self):
        """Test clearing the tracker."""
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Should have data
        assert len(self.tracker.get_steps()) == 1

        # Clear
        self.tracker.clear()

        # Should be empty
        assert len(self.tracker.get_steps()) == 0
        assert self.tracker.get_pca_result(step=0, layer_name="layer_0") is None

    def test_missing_layer_returns_none(self):
        """Test that missing layer returns None."""
        inputs, beliefs, probs, activations = self.create_dummy_data()
        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Query non-existent layer
        result = self.tracker.get_pca_result(step=0, layer_name="nonexistent_layer")
        assert result is None

    def test_missing_step_returns_none(self):
        """Test that missing step returns None."""
        # Query non-existent step
        result = self.tracker.get_pca_result(step=999, layer_name="layer_0")
        assert result is None

    def test_n_pca_components_parameter(self):
        """Test specifying number of PCA components."""
        inputs, beliefs, probs, activations = self.create_dummy_data()

        self.tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
            n_pca_components=3,
        )

        pca_res = self.tracker.get_pca_result(step=0, layer_name="layer_0")
        assert pca_res is not None
        # Should have at most 3 components
        assert pca_res["components"].shape[0] <= 3


class TestAnalysisTrackerEdgeCases:
    def test_single_step_generates_plots(self):
        """Test that plots can be generated with single step."""
        tracker = AnalysisTracker(layer_names=["layer_0"])

        batch_size, seq_len = 5, 4
        n_beliefs = 3
        d_layer = 6

        inputs = jnp.array(np.random.randint(0, 10, size=(batch_size, seq_len)))
        beliefs = jnp.array(np.random.rand(batch_size, seq_len, n_beliefs))
        beliefs = beliefs / beliefs.sum(axis=-1, keepdims=True)
        probs = jnp.array(np.random.rand(batch_size, seq_len))
        activations = {
            "layer_0": jnp.array(np.random.randn(batch_size, seq_len, d_layer))
        }

        tracker.add_step(
            step=0,
            inputs=inputs,
            beliefs=beliefs,
            probs=probs,
            activations_by_layer=activations,
        )

        # Should be able to generate plots
        plots = tracker.generate_pca_plots()
        assert len(plots) > 0

    def test_empty_tracker_generates_no_plots(self):
        """Test that empty tracker generates no plots."""
        tracker = AnalysisTracker(layer_names=["layer_0"])

        plots = tracker.generate_pca_plots()
        assert len(plots) == 0

        plots = tracker.generate_regression_plots()
        assert len(plots) == 0
