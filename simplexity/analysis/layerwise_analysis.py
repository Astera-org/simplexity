"""Composable layer-wise analysis orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from simplexity.analysis.linear_regression import (
    layer_linear_regression,
    layer_linear_regression_svd,
)
from simplexity.analysis.pca import (
    DEFAULT_VARIANCE_THRESHOLDS,
    layer_pca_analysis,
)

AnalysisFn = Callable[..., tuple[Mapping[str, float], Mapping[str, jax.Array]]]


ValidatorFn = Callable[[Mapping[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class AnalysisRegistration:
    """Registry entry describing a supported layer analysis."""

    fn: AnalysisFn
    requires_belief_states: bool
    validator: ValidatorFn


def _validate_linear_regression_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(kwargs or {})
    allowed = {"fit_intercept", "to_factors"}
    unexpected = set(provided) - allowed
    if unexpected:
        raise ValueError(f"Unexpected linear_regression kwargs: {sorted(unexpected)}")
    fit_intercept = bool(provided.get("fit_intercept", True))
    to_factors = bool(provided.get("to_factors", False))
    return {"fit_intercept": fit_intercept, "to_factors": to_factors}


def _validate_linear_regression_svd_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(kwargs or {})
    allowed = {"fit_intercept", "rcond_values", "to_factors"}
    unexpected = set(provided) - allowed
    if unexpected:
        raise ValueError(f"Unexpected linear_regression_svd kwargs: {sorted(unexpected)}")
    fit_intercept = bool(provided.get("fit_intercept", True))
    to_factors = bool(provided.get("to_factors", False))
    rcond_values = provided.get("rcond_values")
    if rcond_values is not None:
        if not isinstance(rcond_values, (list, tuple)):
            raise TypeError("rcond_values must be a sequence of floats")
        if len(rcond_values) == 0:
            raise ValueError("rcond_values must not be empty")
        rcond_values = tuple(float(v) for v in rcond_values)
    return {
        "fit_intercept": fit_intercept,
        "to_factors": to_factors,
        "rcond_values": rcond_values,
    }


def _validate_pca_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(kwargs or {})
    allowed = {"n_components", "variance_thresholds"}
    unexpected = set(provided) - allowed
    if unexpected:
        raise ValueError(f"Unexpected pca kwargs: {sorted(unexpected)}")
    n_components = provided.get("n_components")
    if n_components is not None:
        if not isinstance(n_components, int):
            raise TypeError("n_components must be an int or None")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
    thresholds = provided.get("variance_thresholds", DEFAULT_VARIANCE_THRESHOLDS)
    if not isinstance(thresholds, Sequence):
        raise TypeError("variance_thresholds must be a sequence of floats")
    thresholds_tuple = tuple(float(t) for t in thresholds)
    for threshold in thresholds_tuple:
        if threshold <= 0 or threshold > 1:
            raise ValueError("variance thresholds must be within (0, 1]")
    return {
        "n_components": n_components,
        "variance_thresholds": thresholds_tuple,
    }


ANALYSIS_REGISTRY: dict[str, AnalysisRegistration] = {
    "linear_regression": AnalysisRegistration(
        fn=layer_linear_regression,
        requires_belief_states=True,
        validator=_validate_linear_regression_kwargs,
    ),
    "linear_regression_svd": AnalysisRegistration(
        fn=layer_linear_regression_svd,
        requires_belief_states=True,
        validator=_validate_linear_regression_svd_kwargs,
    ),
    "pca": AnalysisRegistration(
        fn=layer_pca_analysis,
        requires_belief_states=False,
        validator=_validate_pca_kwargs,
    ),
}


class LayerwiseAnalysis:
    """Applies a registered single-layer analysis across an entire network."""

    def __init__(
        self,
        analysis_type: str,
        *,
        last_token_only: bool = False,
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
        skip_first_token: bool = False,
        analysis_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if analysis_type not in ANALYSIS_REGISTRY:
            raise ValueError(f"Unknown analysis_type '{analysis_type}'")
        registration = ANALYSIS_REGISTRY[analysis_type]
        self._analysis_fn = registration.fn
        self._analysis_kwargs = registration.validator(analysis_kwargs)
        self._requires_belief_states = registration.requires_belief_states
        self._last_token_only = last_token_only
        self._concat_layers = concat_layers
        self._use_probs_as_weights = use_probs_as_weights
        self._skip_first_token = skip_first_token

    @property
    def last_token_only(self) -> bool:
        """Whether to use only the last token's activations for analysis."""
        return self._last_token_only

    @property
    def concat_layers(self) -> bool:
        """Whether to concatenate activations from all layers before analysis."""
        return self._concat_layers

    @property
    def use_probs_as_weights(self) -> bool:
        """Whether to use probabilities as weights for analysis."""
        return self._use_probs_as_weights

    @property
    def requires_belief_states(self) -> bool:
        """Whether the analysis needs belief state targets."""
        return self._requires_belief_states

    @property
    def skip_first_token(self) -> bool:
        """Whether to skip the first token (useful for off-manifold initial states)."""
        return self._skip_first_token

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | tuple[jax.Array, ...] | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Analyze activations and return namespaced scalar metrics and projections."""
        if self._requires_belief_states and belief_states is None:
            raise ValueError("This analysis requires belief_states")
        scalars: dict[str, float] = {}
        projections: dict[str, jax.Array] = {}
        for layer_name, layer_activations in activations.items():
            layer_scalars, layer_projections = self._analysis_fn(
                layer_activations,
                weights,
                belief_states,
                **self._analysis_kwargs,
            )
            for key, value in layer_scalars.items():
                scalars[f"{layer_name}_{key}"] = value
            for key, value in layer_projections.items():
                projections[f"{layer_name}_{key}"] = value
        return scalars, projections


__all__ = ["LayerwiseAnalysis", "ANALYSIS_REGISTRY"]
