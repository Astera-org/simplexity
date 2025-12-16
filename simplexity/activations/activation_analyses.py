"""Analysis implementations for Transformer layer activations."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import jax

from simplexity.analysis.layerwise_analysis import LayerwiseAnalysis


class ActivationAnalysis(Protocol):
    """Protocol for activation analysis implementations."""

    @property
    def last_token_only(self) -> bool:
        """Whether to select only the last token for analysis."""
        ...

    @property
    def concat_layers(self) -> bool:
        """Whether to concatenate layer activations for analysis."""
        ...

    @property
    def use_probs_as_weights(self) -> bool:
        """Whether to use probabilities as weights for analysis."""
        ...

    @property
    def skip_first_token(self) -> bool:
        """Whether to skip the first token (useful for off-manifold initial states)."""
        ...

    @property
    def requires_belief_states(self) -> bool:
        """Whether the analysis needs belief state targets."""
        ...

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | tuple[jax.Array, ...] | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Analyze activations and return scalar metrics and projections."""
        ...


class PcaAnalysis(LayerwiseAnalysis):
    """LayerwiseAnalysis wrapper for PCA computations."""

    def __init__(
        self,
        n_components: int | None = None,
        variance_thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99),
        *,
        last_token_only: bool = False,
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
        skip_first_token: bool = False,
    ) -> None:
        analysis_kwargs: dict[str, Any] = {
            "n_components": n_components,
            "variance_thresholds": tuple(variance_thresholds),
        }
        super().__init__(
            analysis_type="pca",
            last_token_only=last_token_only,
            concat_layers=concat_layers,
            use_probs_as_weights=use_probs_as_weights,
            skip_first_token=skip_first_token,
            analysis_kwargs=analysis_kwargs,
        )


class LinearRegressionAnalysis(LayerwiseAnalysis):
    """Weighted linear regression powered by the shared LayerwiseAnalysis registry."""

    def __init__(
        self,
        *,
        last_token_only: bool = False,
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
        skip_first_token: bool = False,
        fit_intercept: bool = True,
        to_factors: bool = False,
    ) -> None:
        super().__init__(
            analysis_type="linear_regression",
            last_token_only=last_token_only,
            concat_layers=concat_layers,
            use_probs_as_weights=use_probs_as_weights,
            skip_first_token=skip_first_token,
            analysis_kwargs={"fit_intercept": fit_intercept, "to_factors": to_factors},
        )


class LinearRegressionSVDAnalysis(LayerwiseAnalysis):
    """LayerwiseAnalysis wrapper for the SVD-based regression implementation."""

    def __init__(
        self,
        *,
        last_token_only: bool = False,
        concat_layers: bool = False,
        use_probs_as_weights: bool = True,
        skip_first_token: bool = False,
        rcond_values: Sequence[float] | None = None,
        fit_intercept: bool = True,
        to_factors: bool = False,
    ) -> None:
        analysis_kwargs: dict[str, Any] = {"fit_intercept": fit_intercept, "to_factors": to_factors}
        if rcond_values is not None:
            analysis_kwargs["rcond_values"] = tuple(rcond_values)
        super().__init__(
            analysis_type="linear_regression_svd",
            last_token_only=last_token_only,
            concat_layers=concat_layers,
            use_probs_as_weights=use_probs_as_weights,
            skip_first_token=skip_first_token,
            analysis_kwargs=analysis_kwargs,
        )
