"""Analysis tracker for collecting metrics across training steps."""

from dataclasses import dataclass, field
from typing import Any

import jax
import pandas as pd
import plotly.graph_objects as go

from simplexity.analysis.data_adapters import (
    pca_results_to_dataframe,
    regression_results_to_dataframe,
    variance_to_dataframe,
)
from simplexity.analysis.pca import (
    plot_components_for_variance_threshold,
)
from simplexity.analysis.plot_configs import (
    generate_cumulative_variance_config,
    generate_pca_2d_config,
    generate_pca_3d_config,
    generate_regression_config,
)
from simplexity.analysis.plot_style_configs import (
    PCA2DStyleConfig,
    PCA3DStyleConfig,
    RegressionStyleConfig,
    VarianceStyleConfig,
)
from simplexity.analysis.regression import (
    RegressionResult,
)
from simplexity.utils.analysis_utils import build_prefix_dataset
from simplexity.visualization.plotly_renderer import build_plotly_figure


@dataclass
class AnalysisTracker:
    """Track analysis metrics across training steps and layers.

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

    layer_names: list[str] | None = None
    variance_thresholds: list[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    use_simple_regression: bool = False  # If True, use sklearn OLS; if False, use k-fold CV

    # Plot styling configurations (can be loaded from YAML via Hydra)
    pca_2d_style: PCA2DStyleConfig | None = None
    pca_3d_style: PCA3DStyleConfig | None = None
    regression_style: RegressionStyleConfig | None = None
    variance_style: VarianceStyleConfig | None = None

    # Internal storage
    _pca_results: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict, init=False)
    _regression_results: dict[int, dict[str, RegressionResult]] = field(default_factory=dict, init=False)
    _variance_threshold_results: dict[int, dict[str, dict[float, int]]] = field(default_factory=dict, init=False)

    def add_step(
        self,
        step: int,
        inputs: jax.Array,
        beliefs: jax.Array,
        probs: jax.Array,
        activations_by_layer: dict[str, jax.Array],
        compute_pca: bool = True,
        compute_regression: bool = True,
        n_pca_components: int | None = None,
        regression_scope: str = "per_token",
        layers: list[str] | None = None,
        analysis_configs: dict | None = None,
        **kwargs,
    ) -> None:
        """Add analysis results for a single training/validation step.

        Args:
            step: training step number
            inputs: (batch, seq_len) token ids
            beliefs: (batch, seq_len, B) beliefs
            probs: (batch, seq_len) probabilities
            activations_by_layer: layer_name -> (batch, seq_len, d) activations
            compute_pca: whether to compute PCA analysis
            compute_regression: whether to compute regression analysis
            n_pca_components: number of PCA components (default: all)
            regression_scope: "per_token" | "final_token" | "sequence_level"
            layers: list of layer names to analyze (None = all layers)
            analysis_configs: dict of analysis configs (auto-extracts scope/layers)
            **kwargs: additional keyword arguments (currently unused)
        """
        # Auto-extract parameters from analysis_configs if provided
        if analysis_configs is not None:
            for cfg in analysis_configs.values():
                if isinstance(cfg, dict) and cfg.get("type") == "regression":
                    if "scope" in cfg:
                        regression_scope = cfg["scope"]
                    if "layers" in cfg:
                        layers = cfg["layers"]
                    break
        # Auto-detect layer names from first call if not provided
        if self.layer_names is None:
            self.layer_names = sorted(activations_by_layer.keys())

        # Filter layers if specified
        layers_to_analyze = layers if layers is not None else self.layer_names

        # Build prefix dataset once (shared by all analyses)
        # Only needed for per_token scope
        prefix_dataset = None
        if regression_scope == "per_token" or compute_pca:
            prefix_dataset = build_prefix_dataset(inputs, beliefs, probs, activations_by_layer)

        self._pca_results[step] = {}
        self._regression_results[step] = {}
        self._variance_threshold_results[step] = {}

        for layer_name in layers_to_analyze:
            # PCA analysis (always uses per_token/prefix-deduplicated data)
            if compute_pca:
                if prefix_dataset is None:
                    raise ValueError("PCA requires prefix_dataset, but it was not initialized")

                from simplexity.analysis.pca import (
                    compute_pca as compute_pca_fn,
                )
                from simplexity.analysis.pca import (
                    compute_variance_thresholds,
                )

                X = prefix_dataset.activations_by_layer[layer_name]
                weights = prefix_dataset.probs

                pca_res = compute_pca_fn(X, n_components=n_pca_components, weights=weights)
                pca_res["beliefs"] = prefix_dataset.beliefs
                self._pca_results[step][layer_name] = pca_res

                # Compute variance thresholds
                threshold_res = compute_variance_thresholds(pca_res, self.variance_thresholds)
                self._variance_threshold_results[step][layer_name] = threshold_res

            # Regression analysis (scope-dependent)
            if compute_regression:
                # Prepare X, Y, weights based on scope
                if regression_scope == "per_token":
                    if prefix_dataset is None:
                        raise ValueError("per_token regression requires prefix_dataset, but it was not initialized")
                    # Use prefix-deduplicated data (current behavior)
                    X = prefix_dataset.activations_by_layer[layer_name]
                    Y = prefix_dataset.beliefs
                    weights = prefix_dataset.probs

                elif regression_scope == "final_token":
                    # Only use final token in each sequence
                    X = activations_by_layer[layer_name][:, -1, :]  # (batch, d)
                    Y = beliefs[:, -1, :]  # (batch, B)
                    weights = probs[:, -1]  # (batch,)

                elif regression_scope == "sequence_level":
                    # Concatenate all activations in sequence
                    batch_size, seq_len, d = activations_by_layer[layer_name].shape
                    X = activations_by_layer[layer_name].reshape(batch_size, seq_len * d)
                    Y = beliefs[:, -1, :]  # Predict final belief
                    weights = probs[:, -1]  # Weight by final token probability

                else:
                    raise ValueError(
                        f"Invalid regression_scope: {regression_scope}. "
                        "Must be 'per_token', 'final_token', or 'sequence_level'"
                    )

                if self.use_simple_regression:
                    # Simple sklearn linear regression (fast, no CV)
                    from simplexity.analysis.regression import regress_simple_sklearn

                    reg_result = regress_simple_sklearn(X, Y, weights)
                else:
                    # K-fold CV with rcond tuning (slower, more robust)
                    import numpy as np

                    from simplexity.analysis.regression import regress_with_kfold_rcond_cv

                    reg_result = regress_with_kfold_rcond_cv(
                        X,
                        Y,
                        weights,
                        rcond_values=[1e-15, 1e-10, 1e-5] + np.logspace(-8, -3, 50).tolist(),
                    )

                self._regression_results[step][layer_name] = reg_result

    def get_steps(self) -> list[int]:
        """Get all recorded steps."""
        return sorted(self._pca_results.keys())

    def get_pca_result(self, step: int, layer_name: str) -> dict[str, Any] | None:
        """Get PCA result for a specific step and layer."""
        return self._pca_results.get(step, {}).get(layer_name)

    def get_regression_result(self, step: int, layer_name: str) -> RegressionResult | None:
        """Get regression result for a specific step and layer."""
        return self._regression_results.get(step, {}).get(layer_name)

    def get_variance_thresholds(self, step: int, layer_name: str) -> dict[float, int] | None:
        """Get variance threshold results for a specific step and layer."""
        return self._variance_threshold_results.get(step, {}).get(layer_name)

    def get_pca_dataframe(self, n_components: int = 2, layers: list[str] | None = None) -> pd.DataFrame:
        """Get PCA results as a DataFrame for visualization.

        Args:
            n_components: Number of principal components (2 for 2D, 3 for 3D)
            layers: Optional list of layer names to include (None = all layers)

        Returns:
            DataFrame with columns: step, layer, point_id, pc1, pc2, [pc3]
        """
        # Filter by layers if specified
        pca_results = self._pca_results
        if layers is not None:
            pca_results = {
                step: {layer: res for layer, res in layer_dict.items() if layer in layers}
                for step, layer_dict in pca_results.items()
            }
        return pca_results_to_dataframe(pca_results, n_components=n_components)

    def get_variance_dataframe(
        self, max_components: int | None = None, layers: list[str] | None = None
    ) -> pd.DataFrame:
        """Get variance explained results as a DataFrame for visualization.

        Args:
            max_components: Maximum number of components to include
            layers: Optional list of layer names to include (None = all layers)

        Returns:
            DataFrame with columns: step, layer, component, cumulative_variance
        """
        # Filter by layers if specified
        pca_results = self._pca_results
        if layers is not None:
            pca_results = {
                step: {layer: res for layer, res in layer_dict.items() if layer in layers}
                for step, layer_dict in pca_results.items()
            }
        return variance_to_dataframe(pca_results, max_components=max_components)

    def get_regression_dataframe(self, layers: list[str] | None = None) -> pd.DataFrame:
        """Get regression results as a DataFrame for visualization.

        Args:
            layers: Optional list of layer names to include (None = all layers)

        Returns:
            DataFrame with columns: step, layer, point_id, x, y, belief_type
        """
        # Filter by layers if specified
        regression_results = self._regression_results
        if layers is not None:
            regression_results = {
                step: {layer: res for layer, res in layer_dict.items() if layer in layers}
                for step, layer_dict in regression_results.items()
            }
        return regression_results_to_dataframe(regression_results)

    def build_data_registry(self, analysis_configs: dict) -> dict[str, pd.DataFrame]:
        """Build data registry from analysis configs.

        Args:
            analysis_configs: Dict of analysis configs with 'name' and 'type' fields

        Returns:
            Dict mapping analysis name to DataFrame
        """
        from typing import cast

        from omegaconf import DictConfig, OmegaConf

        data_registry = {}
        for analysis_cfg in analysis_configs.values():
            # Skip non-dict/non-DictConfig values
            if not isinstance(analysis_cfg, (dict, DictConfig)):
                continue

            # Convert DictConfig to plain dict for type safety
            if isinstance(analysis_cfg, DictConfig):
                converted = OmegaConf.to_container(analysis_cfg, resolve=True)
                if not isinstance(converted, dict):
                    continue
                cfg_dict = cast(dict[str, Any], converted)
            else:
                cfg_dict = analysis_cfg

            # Extract kwargs for the appropriate getter (exclude metadata fields)
            getter_kwargs = {k: v for k, v in cfg_dict.items() if k not in ["name", "type", "use_simple", "scope"]}

            # Call the appropriate getter based on type
            analysis_type = cfg_dict.get("type")
            analysis_name = cfg_dict.get("name")

            if analysis_type == "pca":
                data_registry[analysis_name] = self.get_pca_dataframe(**getter_kwargs)
            elif analysis_type == "variance":
                data_registry[analysis_name] = self.get_variance_dataframe(**getter_kwargs)
            elif analysis_type == "regression":
                data_registry[analysis_name] = self.get_regression_dataframe(**getter_kwargs)

        return data_registry

    def generate_pca_plots(
        self, by_step: bool = True, by_layer: bool = True, combined: bool = True
    ) -> dict[str, go.Figure]:
        """Generate interactive PCA plots using declarative visualization backend.

        Args:
            by_step: deprecated (ignored, only combined plots generated)
            by_layer: deprecated (ignored, only combined plots generated)
            combined: generate combined step+layer plot

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        if combined and self._pca_results:
            # Convert data to DataFrame
            pca_df = pca_results_to_dataframe(self._pca_results, n_components=2)

            # Generate config with optional custom styling
            config = generate_pca_2d_config(
                steps=self.get_steps(),
                layers=self.layer_names,
                style=self.pca_2d_style,
            )

            # Build figure
            fig = build_plotly_figure(config, data_registry={"pca_data": pca_df})
            plots["pca_combined"] = fig

        return plots

    def generate_regression_plots(
        self, by_step: bool = True, by_layer: bool = True, combined: bool = True
    ) -> dict[str, go.Figure]:
        """Generate interactive regression simplex projection plots using declarative backend.

        Args:
            by_step: deprecated (ignored, only combined plots generated)
            by_layer: deprecated (ignored, only combined plots generated)
            combined: generate combined step+layer plot

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        if combined and self._regression_results:
            # Convert data to DataFrame
            reg_df = regression_results_to_dataframe(self._regression_results)

            # Generate config with optional custom styling
            config = generate_regression_config(
                steps=self.get_steps(),
                layers=self.layer_names,
                style=self.regression_style,
            )

            # Build figure
            fig = build_plotly_figure(config, data_registry={"regression_data": reg_df})
            plots["regression_combined"] = fig

        return plots

    def generate_pca_3d_plots(
        self, by_step: bool = True, by_layer: bool = True, combined: bool = True
    ) -> dict[str, go.Figure]:
        """Generate interactive 3D PCA plots using declarative visualization backend.

        Args:
            by_step: deprecated (ignored, only combined plots generated)
            by_layer: deprecated (ignored, only combined plots generated)
            combined: generate combined step+layer plot

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        if combined and self._pca_results:
            # Convert data to DataFrame
            pca_df = pca_results_to_dataframe(self._pca_results, n_components=3)

            # Generate config with optional custom styling
            config = generate_pca_3d_config(
                steps=self.get_steps(),
                layers=self.layer_names,
                style=self.pca_3d_style,
            )

            # Build figure
            fig = build_plotly_figure(config, data_registry={"pca_data": pca_df})
            plots["pca_3d_combined"] = fig

        return plots

    def generate_variance_plots(self, max_components: int | None = 20) -> dict[str, go.Figure]:
        """Generate variance explained plots using declarative visualization backend.

        Args:
            max_components: maximum number of components to display in plots

        Returns:
            dict mapping plot name -> Plotly Figure
        """
        # Return empty if no steps added yet
        if self.layer_names is None:
            return {}

        plots = {}

        # Cumulative variance plots with dropdowns
        if self._pca_results:
            # Use max_components from style config if provided, otherwise use parameter
            _max_components = max_components
            if self.variance_style is not None and self.variance_style.max_components is not None:
                _max_components = self.variance_style.max_components

            # Convert data to DataFrame
            var_df = variance_to_dataframe(self._pca_results, max_components=_max_components)

            # Combined plot with step dropdown (compare layers at each step)
            config_by_step = generate_cumulative_variance_config(
                steps=self.get_steps(),
                layers=self.layer_names,
                style=self.variance_style,
                group_by_step=True,
            )
            fig_by_step = build_plotly_figure(config_by_step, data_registry={"variance_data": var_df})
            # TODO: Add threshold lines using shapes
            plots["cumulative_variance_by_step"] = fig_by_step

            # Combined plot with layer dropdown (compare steps for each layer)
            config_by_layer = generate_cumulative_variance_config(
                steps=self.get_steps(),
                layers=self.layer_names,
                style=self.variance_style,
                group_by_step=False,
            )
            fig_by_layer = build_plotly_figure(config_by_layer, data_registry={"variance_data": var_df})
            # TODO: Add threshold lines using shapes
            plots["cumulative_variance_by_layer"] = fig_by_layer

        # Components required for thresholds over training
        if self._variance_threshold_results:
            fig = plot_components_for_variance_threshold(
                self._variance_threshold_results,
                title="Components Required for Variance Thresholds",
            )
            plots["components_for_thresholds"] = fig

        return plots

    def get_variance_threshold_summary(self) -> dict[str, Any]:
        """Get summary of variance thresholds across all steps and layers.

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
            summary["by_layer"][layer_name] = {threshold: [] for threshold in self.variance_thresholds}
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

    def get_regression_metrics_summary(self) -> dict[str, Any]:
        """Get summary of regression metrics across all steps and layers.

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
                    summary["by_layer"][layer_name]["best_rcond"].append(result.best_rcond)

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

    def save_all_plots(self, output_dir: str) -> dict[str, str]:
        """Save combined plots to HTML files in the output directory.

        Saves the interactive combined plots with step slider and layer dropdown controls,
        plus variance-related plots with threshold markers.

        Args:
            output_dir: directory to save plots (will be created if it doesn't exist)

        Returns:
            dict mapping plot name -> file path

        Saved plots:
            - pca_combined: 2D PCA with step slider + layer dropdown
            - pca_3d_combined: 3D PCA with step slider + layer dropdown
            - regression_combined: Simplex projection with step slider + layer dropdown
            - components_for_thresholds: Components needed for variance thresholds over training
            - cumulative_variance_by_step: CVE with step dropdown (compare layers at each step)
            - cumulative_variance_by_layer: CVE with layer dropdown (compare steps for each layer)
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        saved_plots = {}

        # 2D PCA combined (has step slider + layer dropdown)
        pca_plots = self.generate_pca_plots(by_step=False, by_layer=False, combined=True)
        for name, fig in pca_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        # 3D PCA combined (has step slider + layer dropdown)
        pca_3d_plots = self.generate_pca_3d_plots(by_step=False, by_layer=False, combined=True)
        for name, fig in pca_3d_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        # Regression combined (has step slider + layer dropdown)
        regression_plots = self.generate_regression_plots(by_step=False, by_layer=False, combined=True)
        for name, fig in regression_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        # Variance plots
        variance_plots = self.generate_variance_plots()

        # Save all variance plots (now includes combined dropdown versions)
        for name, fig in variance_plots.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            saved_plots[name] = filepath

        return saved_plots

    def clear(self) -> None:
        """Clear all stored results."""
        self._pca_results.clear()
        self._regression_results.clear()
        self._variance_threshold_results.clear()
