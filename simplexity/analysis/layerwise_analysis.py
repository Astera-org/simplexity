"""Composable layer-wise analysis orchestration."""

# pylint: disable=all # Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all # Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax

from simplexity.analysis.linear_regression import layer_linear_regression, layer_log_probs_regression
from simplexity.analysis.metric_keys import construct_layer_specific_key, format_layer_spec
from simplexity.analysis.pca import (
    DEFAULT_VARIANCE_THRESHOLDS,
    layer_pca_analysis,
)
from simplexity.logger import SIMPLEXITY_LOGGER

AnalysisFn = Callable[..., tuple[Mapping[str, float], Mapping[str, jax.Array]]]


ValidatorFn = Callable[[Mapping[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class AnalysisRegistration:
    """Registry entry describing a supported layer analysis."""

    fn: AnalysisFn
    requires_belief_states: bool
    validator: ValidatorFn
    requires_observation_log_probs: bool = False


def _validate_linear_regression_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(kwargs or {})
    allowed = {"fit_intercept", "concat_belief_states", "compute_subspace_orthogonality", "use_svd", "rcond_values"}
    unexpected = set(provided) - allowed
    if unexpected:
        raise ValueError(f"Unexpected linear_regression kwargs: {sorted(unexpected)}")
    resolved_kwargs = {}
    resolved_kwargs["fit_intercept"] = bool(provided.get("fit_intercept", True))
    resolved_kwargs["concat_belief_states"] = bool(provided.get("concat_belief_states", False))
    resolved_kwargs["compute_subspace_orthogonality"] = bool(provided.get("compute_subspace_orthogonality", False))
    rcond_values = provided.get("rcond_values")
    should_use_svd = rcond_values is not None
    use_svd = bool(provided.get("use_svd", should_use_svd))
    resolved_kwargs["use_svd"] = use_svd
    if use_svd:
        if rcond_values is not None:
            if not isinstance(rcond_values, (list, tuple)):
                raise TypeError("rcond_values must be a sequence of floats")
            if len(rcond_values) == 0:
                raise ValueError("rcond_values must not be empty")
            if not use_svd:
                SIMPLEXITY_LOGGER.warning("rcond_values are only used when use_svd is True")
            rcond_values = tuple(float(v) for v in rcond_values)
        resolved_kwargs["rcond_values"] = rcond_values
    elif rcond_values is not None:
        raise ValueError("rcond_values are only used when use_svd is True")
    return resolved_kwargs


def set_use_svd(
    fn: ValidatorFn,
) -> ValidatorFn:
    """Decorator to set use_svd to True in the kwargs and remove it from output to avoid duplicate with partial."""

    def wrapper(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
        if kwargs and "use_svd" in kwargs and not kwargs["use_svd"]:
            raise ValueError("use_svd cannot be set to False for linear_regression_svd")
        modified_kwargs = dict(kwargs) if kwargs else {}  # Make a copy to avoid mutating the input
        modified_kwargs["use_svd"] = True
        resolved = fn(modified_kwargs)
        resolved.pop("use_svd", None)  # Remove use_svd to avoid duplicate argument with partial
        return resolved

    return wrapper


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


def _validate_log_probs_regression_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(kwargs or {})
    allowed = {"fit_intercept", "use_svd", "rcond_values"}
    unexpected = set(provided) - allowed
    if unexpected:
        raise ValueError(f"Unexpected log_probs_regression kwargs: {sorted(unexpected)}")
    resolved_kwargs = {}
    resolved_kwargs["fit_intercept"] = bool(provided.get("fit_intercept", True))
    rcond_values = provided.get("rcond_values")
    should_use_svd = rcond_values is not None
    use_svd = bool(provided.get("use_svd", should_use_svd))
    resolved_kwargs["use_svd"] = use_svd
    if use_svd:
        if rcond_values is not None:
            if not isinstance(rcond_values, (list, tuple)):
                raise TypeError("rcond_values must be a sequence of floats")
            if len(rcond_values) == 0:
                raise ValueError("rcond_values must not be empty")
            rcond_values = tuple(float(v) for v in rcond_values)
        resolved_kwargs["rcond_values"] = rcond_values
    elif rcond_values is not None:
        raise ValueError("rcond_values are only used when use_svd is True")
    return resolved_kwargs


ANALYSIS_REGISTRY: dict[str, AnalysisRegistration] = {
    "linear_regression": AnalysisRegistration(
        fn=layer_linear_regression,
        requires_belief_states=True,
        validator=_validate_linear_regression_kwargs,
    ),
    "linear_regression_svd": AnalysisRegistration(
        fn=partial(layer_linear_regression, use_svd=True),
        requires_belief_states=True,
        validator=set_use_svd(_validate_linear_regression_kwargs),
    ),
    "pca": AnalysisRegistration(
        fn=layer_pca_analysis,
        requires_belief_states=False,
        validator=_validate_pca_kwargs,
    ),
    "log_probs_regression": AnalysisRegistration(
        fn=layer_log_probs_regression,
        requires_belief_states=False,
        validator=_validate_log_probs_regression_kwargs,
        requires_observation_log_probs=True,
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
        skip_deduplication: bool = False,
        analysis_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if analysis_type not in ANALYSIS_REGISTRY:
            raise ValueError(f"Unknown analysis_type '{analysis_type}'")
        registration = ANALYSIS_REGISTRY[analysis_type]
        self._analysis_fn = registration.fn
        self._analysis_kwargs = registration.validator(analysis_kwargs)
        self._requires_belief_states = registration.requires_belief_states
        self._requires_observation_log_probs = registration.requires_observation_log_probs
        self._last_token_only = last_token_only
        self._concat_layers = concat_layers
        self._use_probs_as_weights = use_probs_as_weights
        self._skip_first_token = skip_first_token
        self._skip_deduplication = skip_deduplication

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
    def requires_observation_log_probs(self) -> bool:
        """Whether the analysis needs observation log probability targets."""
        return self._requires_observation_log_probs

    @property
    def skip_first_token(self) -> bool:
        """Whether to skip the first token (useful for off-manifold initial states)."""
        return self._skip_first_token

    @property
    def skip_deduplication(self) -> bool:
        """Whether to skip prefix/sequence deduplication (faster for large vocabularies)."""
        return self._skip_deduplication

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | tuple[jax.Array, ...] | None = None,
        observation_log_probs: jax.Array | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Analyze activations and return namespaced scalar metrics and arrays."""
        if self._requires_belief_states and belief_states is None:
            raise ValueError("This analysis requires belief_states")
        if self._requires_observation_log_probs and observation_log_probs is None:
            raise ValueError("This analysis requires observation_log_probs")
        scalars: dict[str, float] = {}
        arrays: dict[str, jax.Array] = {}
        for layer_name, layer_activations in activations.items():
            if self._requires_observation_log_probs:
                layer_scalars, layer_arrays = self._analysis_fn(
                    layer_activations,
                    weights,
                    observation_log_probs,
                    **self._analysis_kwargs,
                )
            else:
                layer_scalars, layer_arrays = self._analysis_fn(
                    layer_activations,
                    weights,
                    belief_states,
                    **self._analysis_kwargs,
                )
            formatted_layer_name = format_layer_spec(layer_name)
            for key, value in layer_scalars.items():
                constructed_key = construct_layer_specific_key(key, formatted_layer_name)
                scalars[constructed_key] = value
            for key, value in layer_arrays.items():
                constructed_key = construct_layer_specific_key(key, formatted_layer_name)
                arrays[constructed_key] = value
        return scalars, arrays


__all__ = ["LayerwiseAnalysis", "ANALYSIS_REGISTRY"]
