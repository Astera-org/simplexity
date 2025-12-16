"""Integration tests for projection DataFrame building with factor patterns."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from simplexity.activations.visualization.dataframe_builders import (
    _build_dataframe,
    _build_dataframe_for_mappings,
)
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationDataMapping,
    ActivationVisualizationFieldRef,
    CombinedMappingSection,
)
from simplexity.analysis.linear_regression import layer_linear_regression_svd
from simplexity.exceptions import ConfigValidationError


class TestProjectionDataframeIntegration:
    """Integration tests for projection DataFrame building with factor patterns."""

    def test_factored_projection_dataframe_values_match(self):
        """Test that factored projection values are correctly associated with each factor.

        This is a regression test for the bug where projections looked 'random'
        when visualizing factored linear regression results.
        """
        # Simulate projection keys as produced by LayerwiseAnalysis with to_factors=True
        # Keys format: layer_name_factor_idx/projected
        factor_0_data = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
        factor_1_data = np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])

        projections = {
            "layer_0_factor_0/projected": factor_0_data,
            "layer_0_factor_1/projected": factor_1_data,
        }

        # Metadata columns with 3 samples
        metadata_columns = {
            "step": np.array([3, 3, 3]),
            "sample_index": np.array([0, 1, 2]),
        }

        # Create mappings that use key patterns (as in user's YAML config)
        # Note: Each mapping is for a SPECIFIC component, not a wildcard
        mappings = {
            "factor_*_prob_0": ActivationVisualizationFieldRef(
                source="projections",
                key="factor_*/projected",
                component=0,
                group_as="factor",
            ),
            "factor_*_prob_1": ActivationVisualizationFieldRef(
                source="projections",
                key="factor_*/projected",
                component=1,
                group_as="factor",
            ),
        }

        # Build the DataFrame
        df = _build_dataframe_for_mappings(
            mappings=mappings,
            metadata_columns=metadata_columns,
            projections=projections,
            scalars={},
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        # Verify structure: should have 2 groups (factor_0 and factor_1) * 3 samples = 6 rows
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"

        # Verify factor column exists
        assert "factor" in df.columns, "Missing 'factor' column"

        # Check factor_0 data
        factor_0_rows = df[df["factor"] == "0"]
        assert len(factor_0_rows) == 3, f"Expected 3 rows for factor_0, got {len(factor_0_rows)}"

        # Verify factor_0 prob_0 values match the source data
        np.testing.assert_array_almost_equal(
            np.asarray(factor_0_rows["prob_0"]),
            factor_0_data[:, 0],
            err_msg="Factor 0 prob_0 values don't match source data",
        )

        # Verify factor_0 prob_1 values match the source data
        np.testing.assert_array_almost_equal(
            np.asarray(factor_0_rows["prob_1"]),
            factor_0_data[:, 1],
            err_msg="Factor 0 prob_1 values don't match source data",
        )

        # Check factor_1 data
        factor_1_rows = df[df["factor"] == "1"]
        assert len(factor_1_rows) == 3, f"Expected 3 rows for factor_1, got {len(factor_1_rows)}"

        # Verify factor_1 prob_0 values match the source data
        np.testing.assert_array_almost_equal(
            np.asarray(factor_1_rows["prob_0"]),
            factor_1_data[:, 0],
            err_msg="Factor 1 prob_0 values don't match source data",
        )

        # Verify factor_1 prob_1 values match the source data
        np.testing.assert_array_almost_equal(
            np.asarray(factor_1_rows["prob_1"]),
            factor_1_data[:, 1],
            err_msg="Factor 1 prob_1 values don't match source data",
        )

    def test_factored_projection_different_component_counts(self):
        """Test that factors with different numbers of components are handled correctly.

        Factor 0 has 3 components (states), factor 1 has 2 components.
        Requesting component 2 should work for factor 0 but raise an error for factor 1.
        """
        factor_0_data = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])  # 3 components
        factor_1_data = np.array([[0.5, 0.5], [0.4, 0.6]])  # 2 components

        projections = {
            "layer_0_factor_0/projected": factor_0_data,
            "layer_0_factor_1/projected": factor_1_data,
        }

        metadata_columns = {
            "step": np.array([1, 1]),
            "sample_index": np.array([0, 1]),
        }

        # Request component 2 - this should fail for factor_1 which only has 2 components
        mappings = {
            "factor_*_prob_2": ActivationVisualizationFieldRef(
                source="projections",
                key="factor_*/projected",
                component=2,
                group_as="factor",
            ),
        }

        # Should raise an error because factor_1 doesn't have component 2
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            _build_dataframe_for_mappings(
                mappings=mappings,
                metadata_columns=metadata_columns,
                projections=projections,
                scalars={},
                belief_states=None,
                analysis_concat_layers=False,
                layer_names=["layer_0"],
            )

    def test_combined_projections_and_beliefs_data_integrity(self):
        """Test combined mode with projections and belief states."""
        n_samples = 4
        n_factors = 2
        n_states = 3

        belief_states = np.array(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
                [[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
                [[0.3, 0.4, 0.3], [0.2, 0.6, 0.2]],
                [[0.4, 0.5, 0.1], [0.1, 0.1, 0.8]],
            ]
        )

        noise = np.random.default_rng(42).standard_normal((n_samples, n_factors, n_states)) * 0.01
        projected_values = belief_states + noise

        projections = {
            "layer_0_factor_0/projected": projected_values[:, 0, :],
            "layer_0_factor_1/projected": projected_values[:, 1, :],
        }

        metadata_columns = {
            "analysis": np.array(["test"] * n_samples),
            "step": np.array([10] * n_samples),
            "sample_index": np.arange(n_samples),
        }

        config = ActivationVisualizationConfig(
            name="test_combined",
            backend="altair",
            plot=None,
            data_mapping=ActivationVisualizationDataMapping(
                combined=[
                    CombinedMappingSection(
                        label="prediction",
                        mappings={
                            f"factor_*_prob_{i}": ActivationVisualizationFieldRef(
                                source="projections", key="factor_*/projected", component=i, group_as="factor"
                            )
                            for i in range(n_states)
                        },
                    ),
                    CombinedMappingSection(
                        label="ground_truth",
                        mappings={
                            f"factor_*_prob_{i}": ActivationVisualizationFieldRef(
                                source="belief_states", factor="*", component=i, group_as="factor"
                            )
                            for i in range(n_states)
                        },
                    ),
                ],
                combine_as="data_type",
            ),
        )

        df = _build_dataframe(
            viz_cfg=config,
            metadata_columns=metadata_columns,
            projections=projections,
            scalars={},
            scalar_history={},
            scalar_history_step=None,
            belief_states=belief_states,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        assert "data_type" in df.columns
        assert "factor" in df.columns
        assert len(df) == 2 * n_factors * n_samples

    def test_combined_mode_multiple_layers(self):
        """Test that multiple layers appear correctly in combined mode."""
        n_samples = 3
        n_layers = 4
        n_factors = 2
        n_states = 3

        belief_states = np.random.rand(n_samples, n_factors, n_states)
        projections = {
            f"layer_{layer_idx}_factor_{factor_idx}/projected": np.random.rand(n_samples, n_states)
            for layer_idx in range(n_layers)
            for factor_idx in range(n_factors)
        }

        metadata_columns = {
            "analysis": np.array(["test"] * n_samples),
            "step": np.array([10] * n_samples),
            "sample_index": np.arange(n_samples),
        }

        config = ActivationVisualizationConfig(
            name="test_multilayer",
            backend="altair",
            plot=None,
            data_mapping=ActivationVisualizationDataMapping(
                combined=[
                    CombinedMappingSection(
                        label="prediction",
                        mappings={
                            "factor_*_prob_0": ActivationVisualizationFieldRef(
                                source="projections", key="factor_*/projected", component=0, group_as="factor"
                            ),
                        },
                    ),
                    CombinedMappingSection(
                        label="ground_truth",
                        mappings={
                            "factor_*_prob_0": ActivationVisualizationFieldRef(
                                source="belief_states", factor="*", component=0, group_as="factor"
                            ),
                        },
                    ),
                ],
                combine_as="data_type",
            ),
        )

        df = _build_dataframe(
            viz_cfg=config,
            metadata_columns=metadata_columns,
            projections=projections,
            scalars={},
            scalar_history={},
            scalar_history_step=None,
            belief_states=belief_states,
            analysis_concat_layers=False,
            layer_names=[f"layer_{i}" for i in range(n_layers)],
        )

        pred_df = df[df["data_type"] == "prediction"]
        gt_df = df[df["data_type"] == "ground_truth"]
        assert set(np.unique(np.asarray(pred_df["layer"]))) == {f"layer_{i}" for i in range(n_layers)}
        assert set(np.unique(np.asarray(gt_df["layer"]))) == {"_no_layer_"}

    def test_full_visualization_pipeline_factored_vs_nonfactored(self):
        """Test that factored and non-factored projections produce same results for single factor."""
        projection_data = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        metadata = {"step": np.array([1, 1, 1]), "sample_index": np.arange(3)}

        nf_df = _build_dataframe_for_mappings(
            mappings={"prob_0": ActivationVisualizationFieldRef(source="projections", key="projected", component=0)},
            metadata_columns=metadata,
            projections={"layer_0_projected": projection_data},
            scalars={},
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )
        f_df = _build_dataframe_for_mappings(
            mappings={
                "factor_*_prob_0": ActivationVisualizationFieldRef(
                    source="projections", key="factor_*/projected", component=0, group_as="factor"
                )
            },
            metadata_columns=metadata,
            projections={"layer_0_factor_0/projected": projection_data},
            scalars={},
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        nf_sorted = pd.DataFrame(nf_df).sort_values(by="sample_index")
        f_filtered = pd.DataFrame(f_df[f_df["factor"] == "0"]).sort_values(by="sample_index")
        np.testing.assert_array_almost_equal(
            np.asarray(nf_sorted["prob_0"]),
            np.asarray(f_filtered["prob_0"]),
        )

    def test_linear_regression_projections_match_beliefs(self):
        """Test that linear regression projections closely match original beliefs."""
        n_samples, n_features, n_factors, n_states = 50, 10, 3, 3

        rng = np.random.default_rng(42)
        ds = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        beliefs_combined = ds @ rng.standard_normal((n_features, n_factors * n_states)).astype(np.float32) * 0.1
        beliefs_softmax = np.exp(beliefs_combined.reshape(n_samples, n_factors, n_states))
        beliefs_softmax = beliefs_softmax / beliefs_softmax.sum(axis=2, keepdims=True)

        belief_states = tuple(jnp.array(beliefs_softmax[:, f, :]) for f in range(n_factors))
        scalars, projections = layer_linear_regression_svd(
            jnp.array(ds), jnp.ones(n_samples) / n_samples, belief_states, to_factors=True
        )

        for f in range(n_factors):
            assert scalars[f"factor_{f}/r2"] > 0.8, f"Factor {f} R² too low"
            diff = np.abs(np.asarray(projections[f"factor_{f}/projected"]) - np.asarray(belief_states[f]))
            assert diff.max() < 0.2, f"Factor {f} projections differ too much from beliefs"
