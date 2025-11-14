"""Analysis tracker for collecting metrics across training steps."""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import plotly.graph_objects as go

from simplexity.analysis.pca import (
    compute_pca,
    plot_pca_2d_with_step_slider,
    plot_pca_2d_with_layer_dropdown,
    plot_pca_2d_with_step_and_layer,
    compute_variance_thresholds,
)
from simplexity.analysis.regression import (
    regress_activations_to_beliefs,
    plot_simplex_projection_with_step_slider,
    plot_simplex_projection_with_layer_dropdown,
    plot_simplex_projection_with_step_and_layer,
    RegressionResult,
)
from simplexity.utils.analysis_utils import build_prefix_dataset


@dataclass
class AnalysisTracker:
    """
    Track analysis metrics across training steps and layers.

    Usage:
        # Explicit layer names
        tracker = AnalysisTracker(layer_names=["layer_0", "layer_1", "layer_2"])

        # Or auto-detect all layers
        tracker = AnalysisTracker()

        # During training/validation
        for step in training_steps:
            tracker.add_step(
                step=step,
                inputs=inputs,
                beliefs=beliefs,
                probs=probs,
                activations_by_layer=activations_by_layer,
            )

        # Generate interactive plots
        pca_figs = tracker.generate_pca_plots()
        regression_figs = tracker.generate_regression_plots()

        # Save to disk
        tracker.save_all_plots(output_dir="analysis_plots/")
    """

    layer_names: Optional[List[str]] = None
    variance_thresholds: List[float] = field(
        default_factory=lambda: [0.80, 0.90, 0.95, 0.99]
    )

    # Internal storage
    _pca_results: Dict[int, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict, init=False
    )
    _regression_results: Dict[int, Dict[str, RegressionResult]] = field(
        default_factory=dict, init=False
    )
    _variance_threshold_results: Dict[int, Dict[str, Dict[float, int]]] = field(
        default_factory=dict, init=False
    )

    def add_step(
        self,
        step: int,
        inputs: jax.Array,
        beliefs: jax.Array,
        probs: jax.Array,
        activations_by_layer: Dict[str, jax.Array],
        compute_pca: bool = True,
        compute_regression: bool = True,
        n_pca_components: Optional[int] = None,
    ) -> None:
        """
        Add analysis results for a single training/validation step.

        Args:
            step: training step number
            inputs: (batch, seq_len) token ids
            beliefs: (batch, seq_len, B) beliefs
            probs: (batch, seq_len) probabilities
            activations_by_layer: layer_name -> (batch, seq_len, d) activations
            compute_pca: whether to compute PCA analysis
            compute_regression: whether to compute regression analysis
            n_pca_components: number of PCA components (default: all)
        """
        # Auto-detect layer names from first call if not provided
        if self.layer_names is None:
            self.layer_names = sorted(activations_by_layer.keys())

        # Build prefix dataset once (shared by all analyses)
        prefix_dataset = build_prefix_dataset(
            inputs, beliefs, probs, activations_by_layer
        )

        self._pca_results[step] = {}
        self._regression_results[step] = {}
        self._variance_threshold_results[step] = {}

        for layer_name in self.layer_names:
            X = prefix_dataset.activations_by_layer[layer_name]
            weights = prefix_dataset.probs

            # PCA analysis
            if compute_pca:
                from simplexity.analysis.pca import (
                    compute_pca as compute_pca_fn,
                    compute_variance_thresholds,
                )

                pca_res = compute_pca_fn(
                    X, n_components=n_pca_components, weights=weights
                )
                self._pca_results[step][layer_name] = pca_res

                # Compute variance thresholds
                threshold_res = compute_variance_thresholds(
                    pca_res, self.variance_thresholds
                )
                self._variance_threshold_results[step][layer_name] = threshold_res

            # Regression analysis
            if compute_regression:
                Y = prefix_dataset.beliefs
                from simplexity.analysis.regression import regress_with_kfold_rcond_cv
                import numpy as np

                reg_result = regress_with_kfold_rcond_cv(
                    X,
                    Y,
                    weights,
                    rcond_values=[1e-15, 1e-10, 1e-5] + np.logspace(-8, -3, 50).tolist(),
                )
                self._regression_results[step][layer_name] = reg_result

    def get_steps(self) -> List[int]:
        """Get all recorded steps."""
        return sorted(self._pca_results.keys())

    def get_pca_result(
        self, step: int, layer_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get PCA result for a specific step and layer."""
        return self._pca_results.get(step, {}).get(layer_name)

    def get_regression_result(
        self, step: int, layer_name: str
    ) -> Optional[RegressionResult]:
        """Get regression result for a specific step and layer."""
        return self._regression_results.get(step, {}).get(layer_name)

    def get_variance_thresholds(
        self, step: int, layer_name: str
    ) -> Optional[Dict[float, int]]:
        """Get variance threshold results for a specific step and layer."""
        return self._variance_threshold_results.get(step, {}).get(layer_name)

    def generate_pca_plots(
        self, by_step: bool = True, by_layer: bool = True, combined: bool = True
    ) -> Dict[str, go.Figure]:
        """
        Generate interactive PCA plots.

        Args:
            by_step: generate step slider plots (one per layer)
            by_layer: generate layer dropdown plots (one per step)
            combined: generate combined step+layer plot

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        if by_step:
            # One plot per layer with step slider
            for layer_name in self.layer_names:
                pca_by_step = {
                    step: self._pca_results[step][layer_name]
                    for step in self.get_steps()
                    if layer_name in self._pca_results[step]
                }
                if pca_by_step:
                    fig = plot_pca_2d_with_step_slider(
                        pca_by_step, title=f"PCA: {layer_name} Over Training"
                    )
                    plots[f"pca_step_slider_{layer_name}"] = fig

        if by_layer:
            # One plot per step with layer dropdown
            for step in self.get_steps():
                pca_by_layer = {
                    layer_name: self._pca_results[step][layer_name]
                    for layer_name in self.layer_names
                    if layer_name in self._pca_results[step]
                }
                if pca_by_layer:
                    fig = plot_pca_2d_with_layer_dropdown(
                        pca_by_layer, title=f"PCA: Layers at Step {step}"
                    )
                    plots[f"pca_layer_dropdown_step_{step}"] = fig

        if combined and self._pca_results:
            # Single plot with both step and layer controls
            fig = plot_pca_2d_with_step_and_layer(
                self._pca_results, title="PCA: Training Evolution Across Layers"
            )
            plots["pca_combined"] = fig

        return plots

    def generate_regression_plots(
        self, by_step: bool = True, by_layer: bool = True, combined: bool = True
    ) -> Dict[str, go.Figure]:
        """
        Generate interactive regression simplex projection plots.

        Args:
            by_step: generate step slider plots (one per layer)
            by_layer: generate layer dropdown plots (one per step)
            combined: generate combined step+layer plot

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        if by_step:
            # One plot per layer with step slider
            for layer_name in self.layer_names:
                reg_by_step = {
                    step: self._regression_results[step][layer_name]
                    for step in self.get_steps()
                    if layer_name in self._regression_results[step]
                }
                if reg_by_step:
                    fig = plot_simplex_projection_with_step_slider(
                        reg_by_step, title=f"Regression: {layer_name} Over Training"
                    )
                    plots[f"regression_step_slider_{layer_name}"] = fig

        if by_layer:
            # One plot per step with layer dropdown
            for step in self.get_steps():
                reg_by_layer = {
                    layer_name: self._regression_results[step][layer_name]
                    for layer_name in self.layer_names
                    if layer_name in self._regression_results[step]
                }
                if reg_by_layer:
                    fig = plot_simplex_projection_with_layer_dropdown(
                        reg_by_layer, title=f"Regression: Layers at Step {step}"
                    )
                    plots[f"regression_layer_dropdown_step_{step}"] = fig

        if combined and self._regression_results:
            # Single plot with both step and layer controls
            fig = plot_simplex_projection_with_step_and_layer(
                self._regression_results,
                title="Regression: Training Evolution Across Layers",
            )
            plots["regression_combined"] = fig

        return plots

    def get_variance_threshold_summary(self) -> Dict[str, Any]:
        """
        Get summary of variance thresholds across all steps and layers.

        Returns:
            dict with structure: {
                "by_layer": {layer_name: {threshold: [step1_n_components, step2_n_components, ...]}},
                "by_step": {step: {layer_name: {threshold: n_components}}}
            }
        """
        summary = {"by_layer": {}, "by_step": {}}

        # Return empty if no steps added yet
        if self.layer_names is None:
            return summary

        # By layer
        for layer_name in self.layer_names:
            summary["by_layer"][layer_name] = {
                threshold: [] for threshold in self.variance_thresholds
            }
            for step in self.get_steps():
                thresholds = self.get_variance_thresholds(step, layer_name)
                if thresholds:
                    for threshold, n_components in thresholds.items():
                        summary["by_layer"][layer_name][threshold].append(n_components)

        # By step
        for step in self.get_steps():
            summary["by_step"][step] = {}
            for layer_name in self.layer_names:
                thresholds = self.get_variance_thresholds(step, layer_name)
                if thresholds:
                    summary["by_step"][step][layer_name] = thresholds

        return summary

    def get_regression_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of regression metrics across all steps and layers.

        Returns:
            dict with structure: {
                "by_layer": {layer_name: {"r2": [...], "dist": [...], "best_rcond": [...]}},
                "by_step": {step: {layer_name: {"r2": float, "dist": float, ...}}}
            }
        """
        summary = {"by_layer": {}, "by_step": {}}

        # Return empty if no steps added yet
        if self.layer_names is None:
            return summary

        # By layer
        for layer_name in self.layer_names:
            summary["by_layer"][layer_name] = {
                "r2": [],
                "dist": [],
                "best_rcond": [],
            }
            for step in self.get_steps():
                result = self.get_regression_result(step, layer_name)
                if result:
                    summary["by_layer"][layer_name]["r2"].append(result.r2)
                    summary["by_layer"][layer_name]["dist"].append(result.dist)
                    summary["by_layer"][layer_name]["best_rcond"].append(
                        result.best_rcond
                    )

        # By step
        for step in self.get_steps():
            summary["by_step"][step] = {}
            for layer_name in self.layer_names:
                result = self.get_regression_result(step, layer_name)
                if result:
                    summary["by_step"][step][layer_name] = {
                        "r2": result.r2,
                        "dist": result.dist,
                        "best_rcond": result.best_rcond,
                        "mse": result.mse.tolist(),
                        "mae": result.mae.tolist(),
                        "rmse": result.rmse.tolist(),
                    }

        return summary

    def save_all_plots(self, output_dir: str) -> Dict[str, str]:
        """
        Save all plots to HTML files in the output directory.

        Args:
            output_dir: directory to save plots (will be created if it doesn't exist)

        Returns:
            dict mapping plot name -> file path
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        saved_plots = {}

        # Generate and save PCA plots
        pca_plots = self.generate_pca_plots()
        for name, fig in pca_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        # Generate and save regression plots
        regression_plots = self.generate_regression_plots()
        for name, fig in regression_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        return saved_plots

    def clear(self) -> None:
        """Clear all stored results."""
        self._pca_results.clear()
        self._regression_results.clear()
        self._variance_threshold_results.clear()
