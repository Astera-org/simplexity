"""Tests for the LayerwiseAnalysis orchestrator."""

import jax.numpy as jnp
import pytest

from simplexity.analysis.layerwise_analysis import LayerwiseAnalysis


def _make_synthetic_inputs() -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    activations = {
        "layer_a": jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "layer_b": jnp.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0]]),
    }
    weights = jnp.array([0.2, 0.3, 0.5])
    belief_states = jnp.array([[1.0], [2.0], [3.0]])
    return activations, weights, belief_states


def test_layerwise_analysis_linear_regression_namespacing() -> None:
    """Metrics and projections should be namespace-qualified per layer."""
    activations, weights, belief_states = _make_synthetic_inputs()
    analysis = LayerwiseAnalysis("linear_regression", last_token_only=True)

    scalars, projections = analysis.analyze(
        activations=activations,
        weights=weights,
        belief_states=belief_states,
    )

    assert set(scalars) >= {"layer_a_r2", "layer_b_r2"}
    assert set(projections) == {"layer_a_projected", "layer_b_projected"}


def test_layerwise_analysis_requires_targets() -> None:
    """Analyses that need belief states should validate input."""
    activations, weights, _ = _make_synthetic_inputs()
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


def test_pca_analysis_does_not_require_beliefs() -> None:
    """PCA analysis should run without belief states and namespace results."""
    activations, weights, _ = _make_synthetic_inputs()
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


def test_linear_regression_svd_kwargs_are_normalized() -> None:
    """Validator should coerce mixed numeric types to floats."""
    analysis = LayerwiseAnalysis(
        "linear_regression_svd",
        analysis_kwargs={"rcond_values": [1, 1e-3]},
    )

    assert analysis._analysis_kwargs["rcond_values"] == (1.0, 0.001)


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

    with pytest.raises(ValueError, match="variance thresholds must be within \(0, 1]"):
        LayerwiseAnalysis(
            "pca",
            analysis_kwargs={"variance_thresholds": (0.5, 1.5)},
        )
