"""Tests for activation analysis system."""

import jax.numpy as jnp
import pytest

from simplexity.activations.activation_analyses import LinearRegressionAnalysis, PCAAnalysis
from simplexity.activations.activation_tracker import ActivationTracker, prepare_activations


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
            token_selection="all",
            layer_selection="individual",
        )

        assert "activations" in result
        assert "belief_states" in result
        assert "weights" in result

        assert "layer_0" in result["activations"]
        assert "layer_1" in result["activations"]

        assert result["belief_states"] is not None
        n_prefixes = result["belief_states"].shape[0]
        assert result["activations"]["layer_0"].shape == (n_prefixes, synthetic_data["d_layer0"])
        assert result["activations"]["layer_1"].shape == (n_prefixes, synthetic_data["d_layer1"])
        assert result["belief_states"].shape == (n_prefixes, synthetic_data["belief_dim"])
        assert result["weights"].shape == (n_prefixes,)

    def test_all_tokens_concatenated(self, synthetic_data):
        """Test 'all' tokens with 'concatenated' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="all",
            layer_selection="concatenated",
        )

        assert "concatenated" in result["activations"]
        assert "layer_0" not in result["activations"]
        assert "layer_1" not in result["activations"]

        assert result["belief_states"] is not None
        n_prefixes = result["belief_states"].shape[0]
        expected_d = synthetic_data["d_layer0"] + synthetic_data["d_layer1"]
        assert result["activations"]["concatenated"].shape == (n_prefixes, expected_d)

    def test_last_token_individual(self, synthetic_data):
        """Test 'last' token with 'individual' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
        )

        assert "layer_0" in result["activations"]
        assert "layer_1" in result["activations"]

        assert result["belief_states"] is not None
        batch_size = synthetic_data["batch_size"]
        assert result["activations"]["layer_0"].shape == (batch_size, synthetic_data["d_layer0"])
        assert result["activations"]["layer_1"].shape == (batch_size, synthetic_data["d_layer1"])
        assert result["belief_states"].shape == (batch_size, synthetic_data["belief_dim"])
        assert result["weights"].shape == (batch_size,)

    def test_last_token_concatenated(self, synthetic_data):
        """Test 'last' token with 'concatenated' layers."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="concatenated",
        )

        assert "concatenated" in result["activations"]

        batch_size = synthetic_data["batch_size"]
        expected_d = synthetic_data["d_layer0"] + synthetic_data["d_layer1"]
        assert result["activations"]["concatenated"].shape == (batch_size, expected_d)

    def test_uniform_weights(self, synthetic_data):
        """Test use_probs_as_weights=False produces uniform normalized weights."""
        result = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
            use_probs_as_weights=False,
        )

        # All weights should be equal (uniform)
        assert jnp.allclose(result["weights"], result["weights"][0])
        # Weights should sum to 1
        assert jnp.allclose(result["weights"].sum(), 1.0)

    def test_invalid_token_selection(self, synthetic_data):
        """Test invalid token_selection raises ValueError."""
        with pytest.raises(ValueError, match="Invalid token_selection"):
            prepare_activations(
                synthetic_data["inputs"],
                synthetic_data["beliefs"],
                synthetic_data["probs"],
                synthetic_data["activations"],
                token_selection="invalid",  # type: ignore[arg-type]
                layer_selection="individual",
            )

    def test_invalid_layer_selection(self, synthetic_data):
        """Test invalid layer_selection raises ValueError."""
        with pytest.raises(ValueError, match="Invalid layer_selection"):
            prepare_activations(
                synthetic_data["inputs"],
                synthetic_data["beliefs"],
                synthetic_data["probs"],
                synthetic_data["activations"],
                token_selection="last",
                layer_selection="invalid",  # type: ignore[arg-type]
            )


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
            token_selection="last",
            layer_selection="individual",
        )

        scalars, projections = analysis.analyze(**prepared)

        assert "layer_0_r2" in scalars
        assert "layer_0_rmse" in scalars
        assert "layer_0_mae" in scalars
        assert "layer_0_dist" in scalars
        assert "layer_1_r2" in scalars

        assert "layer_0_projected" in projections
        assert "layer_1_projected" in projections

        assert prepared["belief_states"] is not None
        assert projections["layer_0_projected"].shape == prepared["belief_states"].shape
        assert projections["layer_1_projected"].shape == prepared["belief_states"].shape

    def test_requires_belief_states(self, synthetic_data):
        """Test that analysis raises error without belief_states."""
        analysis = LinearRegressionAnalysis()

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
        )

        prepared["belief_states"] = None

        with pytest.raises(ValueError, match="requires belief_states"):
            analysis.analyze(**prepared)

    def test_uniform_weights(self, synthetic_data):
        """Test regression with uniform weights via use_probs_as_weights=False."""
        analysis = LinearRegressionAnalysis(use_probs_as_weights=False)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
            use_probs_as_weights=False,
        )

        scalars, projections = analysis.analyze(**prepared)

        assert "layer_0_r2" in scalars
        assert "layer_0_projected" in projections


class TestPCAAnalysis:
    """Test PCAAnalysis."""

    def test_basic_pca(self, synthetic_data):
        """Test basic PCA analysis."""
        analysis = PCAAnalysis(n_components=3)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
        )

        scalars, projections = analysis.analyze(**prepared)

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

        batch_size = prepared["activations"]["layer_0"].shape[0]
        assert projections["layer_0_pca"].shape == (batch_size, 3)
        assert projections["layer_1_pca"].shape == (batch_size, 3)

    def test_pca_without_belief_states(self, synthetic_data):
        """Test PCA works without belief_states."""
        analysis = PCAAnalysis(n_components=2)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
        )

        prepared["belief_states"] = None

        scalars, projections = analysis.analyze(**prepared)

        assert "layer_0_cumvar_1" in scalars
        assert "layer_0_cumvar_2" in scalars
        assert "layer_0_pca" in projections

    def test_pca_all_components(self, synthetic_data):
        """Test PCA with n_components=None computes all components."""
        analysis = PCAAnalysis(n_components=None)

        prepared = prepare_activations(
            synthetic_data["inputs"],
            synthetic_data["beliefs"],
            synthetic_data["probs"],
            synthetic_data["activations"],
            token_selection="last",
            layer_selection="individual",
        )

        scalars, projections = analysis.analyze(**prepared)

        batch_size = prepared["activations"]["layer_0"].shape[0]
        d_layer0 = synthetic_data["d_layer0"]
        assert projections["layer_0_pca"].shape == (batch_size, min(batch_size, d_layer0))


class TestActivationTracker:
    """Test ActivationTracker orchestration."""

    def test_basic_tracking(self, synthetic_data):
        """Test basic tracker with multiple analyses."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    token_selection="last",
                    layer_selection="individual",
                ),
                "pca": PCAAnalysis(
                    n_components=2,
                    token_selection="last",
                    layer_selection="individual",
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
                    token_selection="all",
                    layer_selection="individual",
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
                    token_selection="last",
                    layer_selection="individual",
                ),
                "pca": PCAAnalysis(
                    n_components=2,
                    token_selection="last",
                    layer_selection="individual",
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

    def test_concatenated_layers(self, synthetic_data):
        """Test tracker with concatenated layers."""
        tracker = ActivationTracker(
            {
                "regression": LinearRegressionAnalysis(
                    token_selection="last",
                    layer_selection="concatenated",
                ),
                "pca": PCAAnalysis(
                    n_components=2,
                    token_selection="last",
                    layer_selection="concatenated",
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
                    token_selection="last",
                    layer_selection="individual",
                    use_probs_as_weights=False,
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

    def test_multiple_configs_efficiency(self, synthetic_data):
        """Test that tracker efficiently pre-computes only needed preprocessing modes."""
        tracker = ActivationTracker(
            {
                "pca_all_tokens": PCAAnalysis(
                    n_components=2,
                    token_selection="all",
                    layer_selection="individual",
                ),
                "pca_last_token": PCAAnalysis(
                    n_components=3,
                    token_selection="last",
                    layer_selection="individual",
                ),
                "regression_concat": LinearRegressionAnalysis(
                    token_selection="all",
                    layer_selection="concatenated",
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
