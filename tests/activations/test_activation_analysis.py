"""Tests for activation analysis system."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simplexity.activations.activation_analyses import (
    LinearRegressionAnalysis,
    LinearRegressionSVDAnalysis,
    PcaAnalysis,
)
from simplexity.activations.activation_tracker import ActivationTracker, PrepareOptions, prepare_activations
from simplexity.activations.visualization.dataframe_builders import _build_scalar_series_dataframe
from simplexity.activations.visualization_configs import (
    ActivationVisualizationControlsConfig,
    ScalarSeriesMapping,
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
        "batch_size": batch_size,
        "seq_len": seq_len,
        "belief_dim": belief_dim,
        "d_layer0": d_layer0,
        "d_layer1": d_layer1,
    }


class TestPrepareActivations:
    """Test the prepare_activations function."""

    def test_all_tokens_individual(self, synthetic_data):
        """Test 'all' tokens with 'individual' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=False,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert hasattr(result, "activations")
        assert hasattr(result, "belief_states")
        assert hasattr(result, "weights")

        assert "layer_0" in result.activations
        assert "layer_1" in result.activations

        assert result.belief_states is not None
        assert isinstance(result.belief_states, jax.Array)
        n_prefixes = result.belief_states.shape[0]
        assert result.activations["layer_0"].shape == (n_prefixes, synthetic_data["d_layer0"])
        assert result.activations["layer_1"].shape == (n_prefixes, synthetic_data["d_layer1"])
        assert result.belief_states.shape == (n_prefixes, synthetic_data["belief_dim"])
        assert result.weights.shape == (n_prefixes,)

    def test_all_tokens_concatenated(self, synthetic_data):
        """Test 'all' tokens with 'concatenated' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=False,
                concat_layers=True,
                use_probs_as_weights=False,
            ),
        )

        assert "concatenated" in result.activations
        assert "layer_0" not in result.activations
        assert "layer_1" not in result.activations

        assert result.belief_states is not None
        assert isinstance(result.belief_states, jax.Array)
        n_prefixes = result.belief_states.shape[0]
        expected_d = synthetic_data["d_layer0"] + synthetic_data["d_layer1"]
        assert result.activations["concatenated"].shape == (n_prefixes, expected_d)

    def test_last_token_individual(self, synthetic_data):
        """Test 'last' token with 'individual' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert "layer_0" in result.activations
        assert "layer_1" in result.activations

        assert result.belief_states is not None
        assert isinstance(result.belief_states, jax.Array)
        batch_size = synthetic_data["batch_size"]
        assert result.activations["layer_0"].shape == (batch_size, synthetic_data["d_layer0"])
        assert result.activations["layer_1"].shape == (batch_size, synthetic_data["d_layer1"])
        assert result.belief_states.shape == (batch_size, synthetic_data["belief_dim"])
        assert result.weights.shape == (batch_size,)

    def test_last_token_concatenated(self, synthetic_data):
        """Test 'last' token with 'concatenated' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=True,
                use_probs_as_weights=False,
            ),
        )

        assert "concatenated" in result.activations

        batch_size = synthetic_data["batch_size"]
        expected_d = synthetic_data["d_layer0"] + synthetic_data["d_layer1"]
        assert result.activations["concatenated"].shape == (batch_size, expected_d)

    def test_uniform_weights(self, synthetic_data):
        """Test use_probs_as_weights=False produces uniform normalized weights."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        # All weights should be equal (uniform)
        np_weights = np.asarray(result.weights)
        assert np.allclose(np_weights, np_weights[0])
        # Weights should sum to 1
        assert np.allclose(np_weights.sum(), 1.0)

    def test_accepts_torch_inputs(self, synthetic_data):
        """prepare_activations should accept PyTorch tensors."""
        torch = pytest.importorskip("torch")
        inputs = torch.tensor(np.asarray(synthetic_data["inputs"]))
        beliefs = torch.tensor(np.asarray(synthetic_data["beliefs"]))
        probs = torch.tensor(np.asarray(synthetic_data["probs"]))
        activations = {name: torch.tensor(np.asarray(layer)) for name, layer in synthetic_data["activations"].items()}

        result = prepare_activations(
            inputs,
            beliefs,
            probs,
            activations,
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert "layer_0" in result.activations
        assert result.activations["layer_0"].shape[0] == synthetic_data["batch_size"]


class TestLinearRegressionAnalysis:
    """Test LinearRegressionAnalysis."""

    def test_basic_regression(self, synthetic_data):
        """Test basic regression analysis."""
        analysis = LinearRegressionAnalysis()

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        scalars, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        assert "layer_0_r2" in scalars
        assert "layer_0_rmse" in scalars
        assert "layer_0_mae" in scalars
        assert "layer_0_dist" in scalars
        assert "layer_1_r2" in scalars

        assert "layer_0_projected" in projections
        assert "layer_1_projected" in projections

        assert prepared.belief_states is not None
        assert isinstance(prepared.belief_states, jax.Array)
        assert projections["layer_0_projected"].shape == prepared.belief_states.shape
        assert projections["layer_1_projected"].shape == prepared.belief_states.shape

    def test_requires_belief_states(self, synthetic_data):
        """Test that analysis raises error without belief_states."""
        analysis = LinearRegressionAnalysis()

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        prepared.belief_states = None

        with pytest.raises(ValueError, match="requires belief_states"):
            analysis.analyze(
                activations=prepared.activations,
                belief_states=prepared.belief_states,
                weights=prepared.weights,
            )

    def test_uniform_weights(self, synthetic_data):
        """Test regression with uniform weights via use_probs_as_weights=False."""
        analysis = LinearRegressionAnalysis(use_probs_as_weights=False)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        scalars, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        assert "layer_0_r2" in scalars
        assert "layer_0_projected" in projections


class TestLinearRegressionSVDAnalysis:
    """Test LinearRegressionSVDAnalysis."""

    def test_basic_regression_svd(self, synthetic_data):
        """Test SVD regression analysis with rcond tuning."""
        analysis = LinearRegressionSVDAnalysis(rcond_values=[1e-15, 1e-10, 1e-8])

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        scalars, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        assert "layer_0_r2" in scalars
        assert "layer_0_rmse" in scalars
        assert "layer_0_mae" in scalars
        assert "layer_0_dist" in scalars
        assert "layer_0_best_rcond" in scalars
        assert "layer_1_r2" in scalars
        assert "layer_1_best_rcond" in scalars

        assert "layer_0_projected" in projections
        assert "layer_1_projected" in projections

        assert prepared.belief_states is not None
        assert isinstance(prepared.belief_states, jax.Array)
        assert projections["layer_0_projected"].shape == prepared.belief_states.shape
        assert projections["layer_1_projected"].shape == prepared.belief_states.shape

        # Check that best_rcond is one of the provided values
        assert scalars["layer_0_best_rcond"] in [1e-15, 1e-10, 1e-8]

    def test_requires_belief_states(self, synthetic_data):
        """Test that SVD analysis raises error without belief_states."""
        analysis = LinearRegressionSVDAnalysis()

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        prepared.belief_states = None

        with pytest.raises(ValueError, match="requires belief_states"):
            analysis.analyze(
                activations=prepared.activations,
                belief_states=prepared.belief_states,
                weights=prepared.weights,
            )


class TestPcaAnalysis:
    """Test PcaAnalysis."""

    def test_basic_pca(self, synthetic_data):
        """Test basic PCA analysis."""
        analysis = PcaAnalysis(n_components=3)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        scalars, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        assert "layer_0_cumvar_1" in scalars
        assert "layer_0_cumvar_2" in scalars
        assert "layer_0_cumvar_3" in scalars
        assert scalars["layer_0_cumvar_1"] <= scalars["layer_0_cumvar_2"]
        assert scalars["layer_0_cumvar_2"] <= scalars["layer_0_cumvar_3"]
        assert "layer_0_n_components_80pct" in scalars
        assert "layer_0_n_components_90pct" in scalars
        assert "layer_1_cumvar_1" in scalars

        assert "layer_0_pca" in projections
        assert "layer_1_pca" in projections

        batch_size = prepared.activations["layer_0"].shape[0]
        assert projections["layer_0_pca"].shape == (batch_size, 3)
        assert projections["layer_1_pca"].shape == (batch_size, 3)

    def test_pca_without_belief_states(self, synthetic_data):
        """Test PCA works without belief_states."""
        analysis = PcaAnalysis(n_components=2)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        prepared.belief_states = None

        scalars, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        assert "layer_0_cumvar_1" in scalars
        assert "layer_0_cumvar_2" in scalars
        assert "layer_0_pca" in projections

    def test_pca_all_components(self, synthetic_data):
        """Test PCA with n_components=None computes all components."""
        analysis = PcaAnalysis(n_components=None)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        _, projections = analysis.analyze(
            activations=prepared.activations,
            belief_states=prepared.belief_states,
            weights=prepared.weights,
        )

        batch_size = prepared.activations["layer_0"].shape[0]
        d_layer0 = synthetic_data["d_layer0"]
        assert projections["layer_0_pca"].shape == (batch_size, min(batch_size, d_layer0))


class TestActivationTracker:
    """Test ActivationTracker orchestration."""

    def test_basic_tracking(self, synthetic_data):
        """Test basic tracker with multiple analyses."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=True,
                    concat_layers=False,
                ),
                "pca": PcaAnalysis(
                    n_components=2,
                    last_token_only=True,
                    concat_layers=False,
                ),
            }
        )

        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_variance_explained" in scalars

        assert "regression/layer_0_projected" in projections
        assert "pca/layer_0_pca" in projections
        assert visualizations == {}

    def test_all_tokens_mode(self, synthetic_data):
        """Test tracker with all tokens mode."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=False,
                    concat_layers=False,
                ),
            }
        )

        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "regression/layer_0_projected" in projections
        assert visualizations == {}

    def test_mixed_requirements(self, synthetic_data):
        """Test tracker with analyses that have different requirements."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=True,
                    concat_layers=False,
                ),
                "pca": PcaAnalysis(
                    n_components=2,
                    last_token_only=True,
                    concat_layers=False,
                ),
            }
        )

        scalars, _, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_variance_explained" in scalars
        assert visualizations == {}

    def test_concatenated_layers(self, synthetic_data):
        """Test tracker with concatenated layers."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=True,
                    concat_layers=True,
                ),
                "pca": PcaAnalysis(
                    n_components=2,
                    last_token_only=True,
                    concat_layers=True,
                ),
            }
        )

        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/concatenated_r2" in scalars
        assert "pca/concatenated_variance_explained" in scalars

        assert "regression/concatenated_projected" in projections
        assert "pca/concatenated_pca" in projections
        assert visualizations == {}

    def test_uniform_weights(self, synthetic_data):
        """Test tracker with uniform weights."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=True,
                    concat_layers=False,
                    use_probs_as_weights=False,
                ),
            }
        )

        scalars, _, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert visualizations == {}

    def test_multiple_configs_efficiency(self, synthetic_data):
        """Test that tracker efficiently pre-computes only needed preprocessing modes."""
        tracker = ActivationTracker(
            {
                "pca_all_tokens": PcaAnalysis(
                    n_components=2,
                    last_token_only=False,
                    concat_layers=False,
                ),
                "pca_last_token": PcaAnalysis(
                    n_components=3,
                    last_token_only=True,
                    concat_layers=False,
                ),
                "regression_concat": LinearRegressionAnalysis(
                    last_token_only=False,
                    concat_layers=True,
                ),
            }
        )

        scalars, projections, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "pca_all_tokens/layer_0_variance_explained" in scalars
        assert "pca_last_token/layer_0_variance_explained" in scalars
        assert "regression_concat/concatenated_r2" in scalars

        assert "pca_all_tokens/layer_0_pca" in projections
        assert "pca_last_token/layer_0_pca" in projections
        assert "regression_concat/concatenated_projected" in projections
        assert visualizations == {}

    def test_tracker_accepts_torch_inputs(self, synthetic_data):
        """ActivationTracker should handle PyTorch tensors via conversion."""
        torch = pytest.importorskip("torch")
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    last_token_only=True,
                    concat_layers=False,
                ),
                "pca": PcaAnalysis(
                    n_components=2,
                    last_token_only=True,
                    concat_layers=False,
                ),
            }
        )

        torch_inputs = torch.tensor(np.asarray(synthetic_data["inputs"]))
        torch_beliefs = torch.tensor(np.asarray(synthetic_data["beliefs"]))
        torch_probs = torch.tensor(np.asarray(synthetic_data["probs"]))
        torch_activations = {
            name: torch.tensor(np.asarray(layer)) for name, layer in synthetic_data["activations"].items()
        }

        scalars, projections, visualizations = tracker.analyze(
            inputs=torch_inputs,
            beliefs=torch_beliefs,
            probs=torch_probs,
            activations=torch_activations,
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_pca" in projections
        assert visualizations == {}

    def test_tracker_builds_visualizations(self, synthetic_data, monkeypatch):
        """Tracker should build configured visualization payloads."""
        monkeypatch.setattr(
            "simplexity.activations.activation_visualizations.build_altair_chart",
            lambda plot_cfg, registry, controls=None: {
                "backend": "altair",
                "layers": len(plot_cfg.layers),
            },
        )
        monkeypatch.setattr(
            "simplexity.activations.activation_visualizations.build_plotly_figure",
            lambda plot_cfg, registry, controls=None: {
                "backend": "plotly",
                "layers": len(plot_cfg.layers),
            },
        )
        viz_cfg = {
            "name": "pca_projection",
            "data_mapping": {
                "mappings": {
                    "pc0": {"source": "projections", "key": "pca", "component": 0},
                    "belief_state": {"source": "belief_states", "reducer": "argmax"},
                }
            },
            "controls": {"slider": "step", "dropdown": "layer"},
            "layer": {
                "geometry": {"type": "point"},
                "aesthetics": {
                    "x": {"field": "pc0", "type": "quantitative"},
                    "color": {"field": "belief_state", "type": "nominal"},
                },
            },
        }
        tracker = ActivationTracker(
            {
                "pca": PcaAnalysis(
                    n_components=1,
                    last_token_only=False,
                    concat_layers=False,
                ),
            },
            visualizations={"pca": [viz_cfg]},
        )

        _, _, visualizations = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        key = "pca/pca_projection"
        assert key in visualizations
        payload = visualizations[key]
        assert not payload.dataframe.empty
        assert payload.controls is not None
        assert payload.controls.slider is not None
        assert payload.controls.slider.field == "step"
        assert set(payload.dataframe["layer"]) == {"layer_0", "layer_1"}

    def test_controls_accumulate_steps_conflict(self):
        """Controls should forbid accumulate_steps with slider targeting step."""
        with pytest.raises(ConfigValidationError):
            ActivationVisualizationControlsConfig(slider="step", accumulate_steps=True)


class TestScalarSeriesMapping:
    """Tests for scalar_series dataframe construction."""

    def test_infers_indices_when_not_provided(self):
        mapping = ScalarSeriesMapping(
            key_template="{layer}_metric_{index}",
            index_field="component",
            value_field="score",
        )
        metadata_columns = {"step": np.array([0])}
        scalars = {
            "test_analysis/layer_0_metric_1": 0.1,
            "test_analysis/layer_0_metric_2": 0.2,
            "test_analysis/layer_1_metric_1": 0.3,
        }
        df = _build_scalar_series_dataframe(mapping, metadata_columns, scalars, ["layer_0", "layer_1"], "test_analysis")

        assert set(df["component"]) == {1, 2}
        assert set(df[df["layer"] == "layer_0"]["component"]) == {1, 2}
        assert set(df[df["layer"] == "layer_1"]["component"]) == {1}

    def test_infer_indices_errors_when_missing(self):
        mapping = ScalarSeriesMapping(
            key_template="{layer}_metric_{index}",
            index_field="k",
            value_field="value",
        )
        metadata_columns = {"step": np.array([0])}
        scalars = {}

        with pytest.raises(ConfigValidationError):
            _build_scalar_series_dataframe(mapping, metadata_columns, scalars, ["layer_0"], "test_analysis")
