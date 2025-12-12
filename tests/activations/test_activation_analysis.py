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

        scalars, projections = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_variance_explained" in scalars

        assert "regression/layer_0_projected" in projections
        assert "pca/layer_0_pca" in projections

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

        scalars, projections = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "regression/layer_0_projected" in projections

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

        scalars, _ = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_variance_explained" in scalars

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

        scalars, projections = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/concatenated_r2" in scalars
        assert "pca/concatenated_variance_explained" in scalars

        assert "regression/concatenated_projected" in projections
        assert "pca/concatenated_pca" in projections

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

        scalars, _ = tracker.analyze(
            inputs=synthetic_data["inputs"],
            beliefs=synthetic_data["beliefs"],
            probs=synthetic_data["probs"],
            activations=synthetic_data["activations"],
        )

        assert "regression/layer_0_r2" in scalars

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

        scalars, projections = tracker.analyze(
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

        scalars, projections = tracker.analyze(
            inputs=torch_inputs,
            beliefs=torch_beliefs,
            probs=torch_probs,
            activations=torch_activations,
        )

        assert "regression/layer_0_r2" in scalars
        assert "pca/layer_0_pca" in projections


class TestTupleBeliefStates:
    """Test activation tracker with tuple belief states for factored processes."""

    @pytest.fixture
    def factored_belief_data(self):
        """Create synthetic data with factored belief states."""
        batch_size = 4
        seq_len = 5
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

        # Factored beliefs: 2 factors with dimensions 3 and 2
        factor_0 = jnp.ones((batch_size, seq_len, 3)) * 0.3
        factor_1 = jnp.ones((batch_size, seq_len, 2)) * 0.7
        factored_beliefs = (factor_0, factor_1)

        probs = jnp.ones((batch_size, seq_len)) * 0.1

        activations = {
            "layer_0": jnp.ones((batch_size, seq_len, d_layer0)) * 0.3,
            "layer_1": jnp.ones((batch_size, seq_len, d_layer1)) * 0.7,
        }

        return {
            "inputs": inputs,
            "factored_beliefs": factored_beliefs,
            "probs": probs,
            "activations": activations,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "factor_0_dim": 3,
            "factor_1_dim": 2,
            "d_layer0": d_layer0,
            "d_layer1": d_layer1,
        }

    def test_prepare_activations_accepts_tuple_beliefs(self, factored_belief_data):
        """prepare_activations should accept and preserve tuple belief states."""
        result = prepare_activations(
            factored_belief_data["inputs"],
            factored_belief_data["factored_beliefs"],
            factored_belief_data["probs"],
            factored_belief_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 2

        batch_size = factored_belief_data["batch_size"]
        assert result.belief_states[0].shape == (batch_size, factored_belief_data["factor_0_dim"])
        assert result.belief_states[1].shape == (batch_size, factored_belief_data["factor_1_dim"])

    def test_prepare_activations_tuple_beliefs_all_tokens(self, factored_belief_data):
        """Tuple beliefs should work with all tokens mode."""
        result = prepare_activations(
            factored_belief_data["inputs"],
            factored_belief_data["factored_beliefs"],
            factored_belief_data["probs"],
            factored_belief_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=False,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 2

        # With deduplication, we expect fewer samples than batch_size * seq_len
        n_prefixes = result.belief_states[0].shape[0]
        assert result.belief_states[0].shape == (n_prefixes, factored_belief_data["factor_0_dim"])
        assert result.belief_states[1].shape == (n_prefixes, factored_belief_data["factor_1_dim"])
        assert result.activations["layer_0"].shape[0] == n_prefixes

    def test_prepare_activations_torch_tuple_beliefs(self, factored_belief_data):
        """prepare_activations should accept tuple of PyTorch tensors."""
        torch = pytest.importorskip("torch")

        torch_factor_0 = torch.tensor(np.asarray(factored_belief_data["factored_beliefs"][0]))
        torch_factor_1 = torch.tensor(np.asarray(factored_belief_data["factored_beliefs"][1]))
        torch_beliefs = (torch_factor_0, torch_factor_1)

        result = prepare_activations(
            factored_belief_data["inputs"],
            torch_beliefs,
            factored_belief_data["probs"],
            factored_belief_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 2
        # Should be converted to JAX arrays
        assert isinstance(result.belief_states[0], jnp.ndarray)
        assert isinstance(result.belief_states[1], jnp.ndarray)

    def test_prepare_activations_numpy_tuple_beliefs(self, factored_belief_data):
        """prepare_activations should accept tuple of numpy arrays."""
        np_factor_0 = np.asarray(factored_belief_data["factored_beliefs"][0])
        np_factor_1 = np.asarray(factored_belief_data["factored_beliefs"][1])
        np_beliefs = (np_factor_0, np_factor_1)

        result = prepare_activations(
            factored_belief_data["inputs"],
            np_beliefs,
            factored_belief_data["probs"],
            factored_belief_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 2
        # Should be converted to JAX arrays
        assert isinstance(result.belief_states[0], jnp.ndarray)
        assert isinstance(result.belief_states[1], jnp.ndarray)

    def test_linear_regression_with_multiple_factors(self, factored_belief_data):
        """LinearRegressionAnalysis with multi-factor tuple should regress to each factor separately."""
        analysis = LinearRegressionAnalysis()

        prepared = prepare_activations(
            factored_belief_data["inputs"],
            factored_belief_data["factored_beliefs"],
            factored_belief_data["probs"],
            factored_belief_data["activations"],
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

        # Should have separate metrics for each factor
        # Format is: layer_name_factor_idx/metric_name
        assert "layer_0_factor_0/r2" in scalars
        assert "layer_0_factor_1/r2" in scalars
        assert "layer_0_factor_0/rmse" in scalars
        assert "layer_0_factor_1/rmse" in scalars
        assert "layer_0_factor_0/mae" in scalars
        assert "layer_0_factor_1/mae" in scalars
        assert "layer_0_factor_0/dist" in scalars
        assert "layer_0_factor_1/dist" in scalars

        assert "layer_1_factor_0/r2" in scalars
        assert "layer_1_factor_1/r2" in scalars

        # Should have separate projections for each factor
        assert "layer_0_factor_0/projected" in projections
        assert "layer_0_factor_1/projected" in projections
        assert "layer_1_factor_0/projected" in projections
        assert "layer_1_factor_1/projected" in projections

        # Check projection shapes
        batch_size = factored_belief_data["batch_size"]
        assert projections["layer_0_factor_0/projected"].shape == (batch_size, factored_belief_data["factor_0_dim"])
        assert projections["layer_0_factor_1/projected"].shape == (batch_size, factored_belief_data["factor_1_dim"])

    def test_linear_regression_svd_with_multiple_factors(self, factored_belief_data):
        """LinearRegressionSVDAnalysis with multi-factor tuple should regress to each factor separately."""
        analysis = LinearRegressionSVDAnalysis(rcond_values=[1e-10])

        prepared = prepare_activations(
            factored_belief_data["inputs"],
            factored_belief_data["factored_beliefs"],
            factored_belief_data["probs"],
            factored_belief_data["activations"],
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

        # Should have separate metrics for each factor including best_rcond
        assert "layer_0_factor_0/r2" in scalars
        assert "layer_0_factor_1/r2" in scalars
        assert "layer_0_factor_0/best_rcond" in scalars
        assert "layer_0_factor_1/best_rcond" in scalars

        # Should have separate projections for each factor
        assert "layer_0_factor_0/projected" in projections
        assert "layer_0_factor_1/projected" in projections

    def test_tracker_with_factored_beliefs(self, factored_belief_data):
        """ActivationTracker should work with tuple belief states."""
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

        scalars, projections = tracker.analyze(
            inputs=factored_belief_data["inputs"],
            beliefs=factored_belief_data["factored_beliefs"],
            probs=factored_belief_data["probs"],
            activations=factored_belief_data["activations"],
        )

        # Regression should have per-factor metrics
        assert "regression/layer_0_factor_0/r2" in scalars
        assert "regression/layer_0_factor_1/r2" in scalars

        # PCA should still work (doesn't use belief states)
        assert "pca/layer_0_variance_explained" in scalars

        # Projections should be present
        assert "regression/layer_0_factor_0/projected" in projections
        assert "regression/layer_0_factor_1/projected" in projections
        assert "pca/layer_0_pca" in projections

    def test_single_factor_tuple(self, synthetic_data):
        """Test with a single-factor tuple (edge case)."""
        # Create single-factor tuple
        single_factor = (synthetic_data["beliefs"],)

        result = prepare_activations(
            synthetic_data["inputs"],
            single_factor,
            synthetic_data["probs"],
            synthetic_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 1
        assert result.belief_states[0].shape == (synthetic_data["batch_size"], synthetic_data["belief_dim"])

    def test_linear_regression_single_factor_tuple_behaves_like_non_tuple(self, synthetic_data):
        """LinearRegressionAnalysis with single-factor tuple should behave like non-tuple (no factor keys)."""
        single_factor = (synthetic_data["beliefs"],)
        analysis = LinearRegressionAnalysis()

        prepared = prepare_activations(
            synthetic_data["inputs"],
            single_factor,
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

        # Should have simple keys without "factor_" prefix
        assert "layer_0_r2" in scalars
        assert "layer_0_rmse" in scalars
        assert "layer_0_projected" in projections

        # Should NOT have factor keys
        assert "layer_0_factor_0/r2" not in scalars
        assert "layer_0_factor_0/projected" not in projections

    def test_linear_regression_concat_belief_states(self, factored_belief_data):
        """LinearRegressionAnalysis with concat_belief_states=True should return both factor and concat results."""
        analysis = LinearRegressionAnalysis(concat_belief_states=True)

        prepared = prepare_activations(
            factored_belief_data["inputs"],
            factored_belief_data["factored_beliefs"],
            factored_belief_data["probs"],
            factored_belief_data["activations"],
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

        # Should have per-factor results
        assert "layer_0_factor_0/r2" in scalars
        assert "layer_0_factor_1/r2" in scalars
        assert "layer_0_factor_0/projected" in projections
        assert "layer_0_factor_1/projected" in projections

        # Should ALSO have concatenated results
        assert "layer_0_concat/r2" in scalars
        assert "layer_0_concat/rmse" in scalars
        assert "layer_0_concat/projected" in projections

        # Check concatenated projection shape (should be sum of factor dimensions)
        batch_size = factored_belief_data["batch_size"]
        total_dim = factored_belief_data["factor_0_dim"] + factored_belief_data["factor_1_dim"]
        assert projections["layer_0_concat/projected"].shape == (batch_size, total_dim)

    def test_three_factor_tuple(self, factored_belief_data):
        """Test with three factors to ensure generalization."""
        batch_size = factored_belief_data["batch_size"]
        seq_len = factored_belief_data["seq_len"]

        # Add a third factor
        factor_0 = jnp.ones((batch_size, seq_len, 3)) * 0.3
        factor_1 = jnp.ones((batch_size, seq_len, 2)) * 0.5
        factor_2 = jnp.ones((batch_size, seq_len, 4)) * 0.7
        three_factor_beliefs = (factor_0, factor_1, factor_2)

        result = prepare_activations(
            factored_belief_data["inputs"],
            three_factor_beliefs,
            factored_belief_data["probs"],
            factored_belief_data["activations"],
            prepare_options=PrepareOptions(
                last_token_only=True,
                concat_layers=False,
                use_probs_as_weights=False,
            ),
        )

        assert result.belief_states is not None
        assert isinstance(result.belief_states, tuple)
        assert len(result.belief_states) == 3
        assert result.belief_states[0].shape == (batch_size, 3)
        assert result.belief_states[1].shape == (batch_size, 2)
        assert result.belief_states[2].shape == (batch_size, 4)
