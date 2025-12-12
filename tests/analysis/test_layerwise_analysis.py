"""Tests for the LayerwiseAnalysis orchestrator."""

import jax
import jax.numpy as jnp
import pytest

from simplexity.analysis.layerwise_analysis import ANALYSIS_REGISTRY, LayerwiseAnalysis


@pytest.fixture
def analysis_inputs() -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Provides sample activations, weights, and belief states for analysis tests."""

    activations = {
        "layer_a": jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "layer_b": jnp.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0]]),
    }
    weights = jnp.array([0.2, 0.3, 0.5])
    belief_states = jnp.array([[1.0], [2.0], [3.0]])
    return activations, weights, belief_states


def test_layerwise_analysis_linear_regression_namespacing(analysis_inputs) -> None:
    """Metrics and projections should be namespace-qualified per layer."""

    activations, weights, belief_states = analysis_inputs
    analysis = LayerwiseAnalysis("linear_regression", last_token_only=True)

    scalars, projections = analysis.analyze(
        activations=activations,
        weights=weights,
        belief_states=belief_states,
    )

    assert set(scalars) >= {"layer_a_r2", "layer_b_r2"}
    assert set(projections) == {
        "layer_a_projected",
        "layer_b_projected",
        "layer_a_coeffs",
        "layer_b_coeffs",
        "layer_a_intercept",
        "layer_b_intercept",
    }


def test_layerwise_analysis_requires_targets(analysis_inputs) -> None:
    """Analyses that need belief states should validate input."""

    activations, weights, _ = analysis_inputs
    analysis = LayerwiseAnalysis("linear_regression")

    with pytest.raises(ValueError, match="requires belief_states"):
        analysis.analyze(activations=activations, weights=weights, belief_states=None)


def test_invalid_analysis_type_raises() -> None:
    """Unknown analysis types should raise clear errors."""

    with pytest.raises(ValueError, match="Unknown analysis_type"):
        LayerwiseAnalysis("unknown")


def test_invalid_kwargs_validation() -> None:
    """Validator rejects unsupported kwargs for a registered analysis."""

    with pytest.raises(ValueError, match="Unexpected linear_regression kwargs"):
        LayerwiseAnalysis(
            "linear_regression",
            analysis_kwargs={"bad": True},
        )


def test_pca_analysis_does_not_require_beliefs(analysis_inputs) -> None:
    """PCA analysis should run without belief states and namespace results."""

    activations, weights, _ = analysis_inputs
    analysis = LayerwiseAnalysis(
        "pca",
        analysis_kwargs={"n_components": 2, "variance_thresholds": (0.5,)},
    )
    scalars, projections = analysis.analyze(
        activations=activations,
        weights=weights,
        belief_states=None,
    )
    assert "layer_a_cumvar_1" in scalars
    assert "layer_a_n_components_50pct" in scalars
    assert "layer_a_pca" in projections


def test_invalid_pca_kwargs() -> None:
    """Invalid PCA kwargs should raise helpful errors."""

    with pytest.raises(ValueError, match="n_components must be positive"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"n_components": 0},
        )


def test_linear_regression_svd_kwargs_validation_errors() -> None:
    """SVD-specific validators should reject unsupported inputs."""

    with pytest.raises(TypeError, match="rcond_values must be a sequence"):
        LayerwiseAnalysis(
            "linear_regression_svd",
            analysis_kwargs={"rcond_values": 0.1},
        )

    with pytest.raises(ValueError, match="rcond_values must not be empty"):
        LayerwiseAnalysis(
            "linear_regression_svd",
            analysis_kwargs={"rcond_values": []},
        )


def test_linear_regression_svd_rejects_unexpected_kwargs() -> None:
    """Unexpected SVD kwargs should raise clear errors."""

    with pytest.raises(ValueError, match="Unexpected linear_regression kwargs"):
        LayerwiseAnalysis(
            "linear_regression_svd",
            analysis_kwargs={"bad": True},
        )


def test_linear_regression_svd_kwargs_are_normalized() -> None:
    """Validator should coerce mixed numeric types to floats."""

    validator = ANALYSIS_REGISTRY["linear_regression_svd"].validator
    params = validator({"rcond_values": [1, 1e-3]})

    assert params["rcond_values"] == (1.0, 0.001)


def test_pca_kwargs_require_int_components() -> None:
    """PCA validator should enforce integral n_components."""

    with pytest.raises(TypeError, match="n_components must be an int or None"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"n_components": 1.5},
        )


def test_pca_kwargs_require_sequence_thresholds() -> None:
    """Variance thresholds must be sequences with valid ranges."""

    with pytest.raises(TypeError, match="variance_thresholds must be a sequence"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"variance_thresholds": 0.9},
        )

    with pytest.raises(ValueError, match=r"variance thresholds must be within \(0, 1]"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"variance_thresholds": (0.5, 1.5)},
        )


def test_pca_rejects_unexpected_kwargs() -> None:
    """Unexpected PCA kwargs should surface informative errors."""

    with pytest.raises(ValueError, match="Unexpected pca kwargs"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"bad": True},
        )


def test_layerwise_analysis_property_accessors() -> None:
    """Constructor flags should surface via property accessors."""

    analysis = LayerwiseAnalysis(
        "pca",
        last_token_only=True,
        concat_layers=True,
        use_probs_as_weights=False,
    )

    assert analysis.last_token_only
    assert analysis.concat_layers
    assert not analysis.use_probs_as_weights
    assert not analysis.requires_belief_states


def test_linear_regression_accepts_concat_belief_states() -> None:
    """linear_regression validator should accept concat_belief_states parameter."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({"fit_intercept": False, "concat_belief_states": True})

    assert params["fit_intercept"] is False
    assert params["concat_belief_states"] is True


def test_linear_regression_svd_accepts_concat_belief_states() -> None:
    """linear_regression_svd validator should accept concat_belief_states parameter."""

    validator = ANALYSIS_REGISTRY["linear_regression_svd"].validator
    params = validator({"fit_intercept": True, "concat_belief_states": True, "rcond_values": [1e-3]})

    assert params["fit_intercept"] is True
    assert params["concat_belief_states"] is True
    assert params["rcond_values"] == (0.001,)


def test_linear_regression_concat_belief_states_defaults_false() -> None:
    """concat_belief_states should default to False when not provided."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({"fit_intercept": True})

    assert params["concat_belief_states"] is False


def test_linear_regression_accepts_compute_subspace_orthogonality() -> None:
    """linear_regression validator should accept compute_subspace_orthogonality parameter."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({"fit_intercept": True, "compute_subspace_orthogonality": True})

    assert params["fit_intercept"] is True
    assert params["compute_subspace_orthogonality"] is True


def test_linear_regression_svd_accepts_compute_subspace_orthogonality() -> None:
    """linear_regression_svd validator should accept compute_subspace_orthogonality parameter."""

    validator = ANALYSIS_REGISTRY["linear_regression_svd"].validator
    params = validator({"fit_intercept": True, "compute_subspace_orthogonality": True, "rcond_values": [1e-3]})

    assert params["fit_intercept"] is True
    assert params["compute_subspace_orthogonality"] is True
    assert params["rcond_values"] == (0.001,)


def test_linear_regression_compute_subspace_orthogonality_defaults_false() -> None:
    """compute_subspace_orthogonality should default to False when not provided."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({"fit_intercept": True})

    assert params["compute_subspace_orthogonality"] is False


def test_linear_regression_accepts_use_svd() -> None:
    """linear_regression validator should accept use_svd parameter."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({"use_svd": True})

    assert params["use_svd"] is True


def test_linear_regression_use_svd_defaults_false() -> None:
    """use_svd should default to False when not provided."""

    validator = ANALYSIS_REGISTRY["linear_regression"].validator
    params = validator({})

    assert params["use_svd"] is False
