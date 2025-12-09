"""Tests for step-wise scalar history tracking in ActivationTracker."""

import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.activations.activation_analyses import PcaAnalysis
from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.activations.activation_visualizations import _build_dataframe
from simplexity.activations.visualization_configs import (
    ActivationVisualizationConfig,
    ActivationVisualizationDataMapping,
    ActivationVisualizationFieldRef,
)
from simplexity.exceptions import ConfigValidationError


@pytest.fixture
def synthetic_data():
    """Create synthetic data for testing."""
    batch_size = 4
    seq_len = 5
    belief_dim = 3
    d_layer0 = 8
    d_layer1 = 12

    inputs = jnp.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [1, 2, 8, 9, 10],
            [1, 2, 3, 4, 11],
        ]
    )

    beliefs = jnp.ones((batch_size, seq_len, belief_dim)) * 0.5
    probs = jnp.ones((batch_size, seq_len)) * 0.1

    activations = {
        "layer_0": jnp.ones((batch_size, seq_len, d_layer0)) * 0.3,
        "layer_1": jnp.ones((batch_size, seq_len, d_layer1)) * 0.7,
    }

    return {
        "inputs": inputs,
        "beliefs": beliefs,
        "probs": probs,
        "activations": activations,
    }


class TestScalarHistory:
    """Tests for scalar history tracking functionality."""

    def test_scalar_history_empty_on_init(self):
        """Scalar history should be empty when tracker is initialized."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        df = tracker.get_scalar_history_df()
        assert df.empty
        assert list(df.columns) == ["metric", "step", "value"]

    def test_scalar_history_without_step_parameter(self, synthetic_data):
        """Calling analyze without step parameter should not accumulate history."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        scalars, _, _ = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert scalars  # Scalars should still be returned
        df = tracker.get_scalar_history_df()
        assert df.empty  # But history should remain empty

    def test_scalar_history_single_step(self, synthetic_data):
        """Scalar history should record values when step is provided."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        scalars, _, _ = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=0,
        )

        df = tracker.get_scalar_history_df()
        assert not df.empty
        assert list(df.columns) == ["metric", "step", "value"]
        assert len(df) == len(scalars)
        assert all(df["step"] == 0)

        for metric_name, scalar_value in scalars.items():
            metric_rows = df[df["metric"] == metric_name]
            assert len(metric_rows) == 1
            assert metric_rows.iloc[0]["value"] == float(scalar_value)

    def test_scalar_history_multiple_steps(self, synthetic_data):
        """Scalar history should accumulate across multiple steps."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        steps = [0, 10, 20, 30]
        for step in steps:
            tracker.analyze(
                inputs=synthetic_data["inputs"],
                beliefs=synthetic_data["beliefs"],
                probs=synthetic_data["probs"],
                activations=synthetic_data["activations"],
                step=step,
            )

        df = tracker.get_scalar_history_df()
        assert not df.empty
        assert df["step"].nunique() == len(steps)
        assert sorted(df["step"].unique()) == steps

        # Each scalar metric should have one entry per step
        for metric_name in df["metric"].unique():
            metric_rows = df[df["metric"] == metric_name]
            assert len(metric_rows) == len(steps)

    def test_scalar_history_mixed_with_and_without_step(self, synthetic_data):
        """Mixing calls with and without step parameter should work correctly."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        # Call without step
        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        df = tracker.get_scalar_history_df()
        assert df.empty

        # Call with step
        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=5,
        )

        df = tracker.get_scalar_history_df()
        assert not df.empty
        assert all(df["step"] == 5)

        # Call without step again
        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        df = tracker.get_scalar_history_df()
        assert not df.empty
        assert all(df["step"] == 5)  # History should still only contain step 5

    def test_scalar_history_multiple_analyses(self, synthetic_data):
        """Scalar history should track metrics from multiple analyses separately."""
        pca_analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        pca_analysis2 = PcaAnalysis(n_components=5, last_token_only=False, concat_layers=False)

        tracker = ActivationTracker({"pca": pca_analysis, "pca_alt": pca_analysis2})

        tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
            step=0,
        )

        df = tracker.get_scalar_history_df()
        assert not df.empty

        # Check that metrics from both analyses are present
        pca_metrics = df[df["metric"].str.startswith("pca/")]
        pca_alt_metrics = df[df["metric"].str.startswith("pca_alt/")]

        assert len(pca_metrics) > 0
        assert len(pca_alt_metrics) > 0

    def test_scalar_history_dataframe_structure(self, synthetic_data):
        """Validate the structure of the scalar history DataFrame."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        for step in [0, 1, 2]:
            tracker.analyze(
                inputs=synthetic_data["inputs"],
                beliefs=synthetic_data["beliefs"],
                probs=synthetic_data["probs"],
                activations=synthetic_data["activations"],
                step=step,
            )

        df = tracker.get_scalar_history_df()

        # Check column types
        assert df["metric"].dtype == object
        assert df["step"].dtype in [int, "int64", "int32"]
        assert df["value"].dtype in [float, "float64", "float32"]

        # Check no null values
        assert not df["metric"].isnull().any()
        assert not df["step"].isnull().any()
        assert not df["value"].isnull().any()

    def test_scalar_history_preserves_order(self, synthetic_data):
        """Scalar history should preserve the order of steps."""
        analysis = PcaAnalysis(n_components=3, last_token_only=True, concat_layers=False)
        tracker = ActivationTracker({"pca": analysis})

        # Add steps in non-sequential order
        steps = [5, 1, 10, 3, 7]
        for step in steps:
            tracker.analyze(
                inputs=synthetic_data["inputs"],
                beliefs=synthetic_data["beliefs"],
                probs=synthetic_data["probs"],
                activations=synthetic_data["activations"],
                step=step,
            )

        df = tracker.get_scalar_history_df()

        # For each metric, steps should appear in the order they were added
        for metric_name in df["metric"].unique():
            metric_df = df[df["metric"] == metric_name]
            recorded_steps = metric_df["step"].tolist()
            assert recorded_steps == steps


class TestScalarHistoryVisualizations:
    """Ensure scalar_history visualizations leverage step-aware accumulation."""

    def _viz_cfg(self) -> ActivationVisualizationConfig:
        return ActivationVisualizationConfig(
            name="history",
            data_mapping=ActivationVisualizationDataMapping(
                mappings={
                    "rmse": ActivationVisualizationFieldRef(
                        source="scalar_pattern",
                        key="layer_*_rmse",
                    )
                }
            ),
        )

    def _metadata(self) -> dict[str, np.ndarray]:
        return {
            "analysis": np.asarray(["analysis"], dtype=object),
            "step": np.asarray([1], dtype=np.int32),
        }

    def test_scalar_history_dataframe_requires_step(self):
        viz_cfg = self._viz_cfg()
        with pytest.raises(
            ConfigValidationError,
            match=r"scalar_pattern/scalar_history source but analyze\(\) was called without the `step` parameter",
        ):
            _build_dataframe(
                viz_cfg,
                self._metadata(),
                projections={},
                scalars={"analysis/layer_0_rmse": 0.1},
                scalar_history={},
                scalar_history_step=None,
                belief_states=None,
                analysis_concat_layers=False,
                layer_names=["layer_0"],
            )

    def test_scalar_history_dataframe_uses_current_step(self):
        viz_cfg = self._viz_cfg()
        df = _build_dataframe(
            viz_cfg,
            self._metadata(),
            projections={},
            scalars={"analysis/layer_0_rmse": 0.42},
            scalar_history={},
            scalar_history_step=7,
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        assert list(df["step"]) == [7]
        assert list(df["layer"]) == ["layer_0"]
        assert list(df["rmse"]) == [0.42]
        assert list(df["metric"]) == ["analysis/layer_0_rmse"]

    def test_scalar_history_pattern_matches_complex_keys(self):
        viz_cfg = ActivationVisualizationConfig(
            name="history",
            data_mapping=ActivationVisualizationDataMapping(
                mappings={
                    "rmse": ActivationVisualizationFieldRef(
                        source="scalar_pattern",
                        key="blocks.*.hook_resid_*_rmse",
                    )
                }
            ),
        )
        scalars = {
            "analysis/blocks.0.hook_resid_pre_rmse": 0.1,
            "analysis/blocks.0.hook_resid_mid_rmse": 0.2,
            "analysis/blocks.1.hook_resid_post_rmse": 0.3,
        }

        df = _build_dataframe(
            viz_cfg,
            self._metadata(),
            projections={},
            scalars=scalars,
            scalar_history={},
            scalar_history_step=11,
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        sorted_rows = df.sort_values("metric").reset_index(drop=True)
        ordered_keys = sorted(scalars.keys())
        assert list(sorted_rows["metric"]) == ordered_keys
        assert list(sorted_rows["rmse"]) == [scalars[key] for key in ordered_keys]
        assert sorted(df["layer"].unique()) == [
            "blocks.0.hook_resid_mid_rmse",
            "blocks.0.hook_resid_pre_rmse",
            "blocks.1.hook_resid_post_rmse",
        ]

    def test_scalar_history_pattern_matches_without_prefix(self):
        viz_cfg = ActivationVisualizationConfig(
            name="history",
            data_mapping=ActivationVisualizationDataMapping(
                mappings={
                    "rmse": ActivationVisualizationFieldRef(
                        source="scalar_pattern",
                        key="blocks.*.hook_resid_*_rmse",
                    )
                }
            ),
        )
        scalars = {
            "blocks.0.hook_resid_pre_rmse": 0.5,
            "blocks.1.hook_resid_pre_rmse": 0.6,
        }

        df = _build_dataframe(
            viz_cfg,
            self._metadata(),
            projections={},
            scalars=scalars,
            scalar_history={},
            scalar_history_step=3,
            belief_states=None,
            analysis_concat_layers=False,
            layer_names=["layer_0"],
        )

        assert sorted(df["metric"]) == sorted(scalars.keys())

    def test_scalar_history_pattern_requires_match(self):
        viz_cfg = ActivationVisualizationConfig(
            name="history",
            data_mapping=ActivationVisualizationDataMapping(
                mappings={
                    "rmse": ActivationVisualizationFieldRef(
                        source="scalar_pattern",
                        key="blocks.*.hook_resid_*_rmse",
                    )
                }
            ),
        )

        with pytest.raises(ConfigValidationError, match="No scalar pattern keys found matching pattern"):
            _build_dataframe(
                viz_cfg,
                self._metadata(),
                projections={},
                scalars={"analysis/other_metric": 0.1},
                scalar_history={},
                scalar_history_step=0,
                belief_states=None,
                analysis_concat_layers=False,
                layer_names=["layer_0"],
            )
