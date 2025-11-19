"""Activation analysis for Transformer layers."""

from collections.abc import Mapping
from typing import Protocol, TypedDict

import jax

from simplexity.utils.analysis_utils import build_last_token_dataset, build_prefix_dataset


class PreparedActivations(TypedDict):
    """Prepared activations with belief states and sample weights."""

    activations: Mapping[str, jax.Array]
    belief_states: jax.Array | None
    weights: jax.Array


def prepare_activations(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations: Mapping[str, jax.Array],
    token_selection: str,
    layer_selection: str,
    use_probs_as_weights: bool = True,
) -> PreparedActivations:
    """Preprocess activations by deduplicating sequences, selecting tokens/layers, and computing weights."""
    if token_selection == "all":
        prefix_dataset = build_prefix_dataset(inputs, beliefs, probs, dict(activations))
        belief_states = prefix_dataset.beliefs
        layer_acts = prefix_dataset.activations_by_layer

        if use_probs_as_weights:
            weights = prefix_dataset.probs
        else:
            import jax.numpy as jnp

            weights = jnp.ones(belief_states.shape[0], dtype=beliefs.dtype)
            weights = weights / weights.sum()

    elif token_selection == "last":
        last_beliefs = beliefs[:, -1, :]
        last_probs = probs[:, -1]
        last_layer_acts = {name: acts[:, -1, :] for name, acts in activations.items()}

        last_token_dataset = build_last_token_dataset(inputs, last_beliefs, last_probs, last_layer_acts)
        belief_states = last_token_dataset.beliefs
        layer_acts = last_token_dataset.activations_by_layer

        if use_probs_as_weights:
            weights = last_token_dataset.probs
        else:
            import jax.numpy as jnp

            weights = jnp.ones(belief_states.shape[0], dtype=beliefs.dtype)
            weights = weights / weights.sum()

    else:
        raise ValueError(f"Invalid token_selection: {token_selection}. Must be 'all' or 'last'.")

    if layer_selection == "concatenated":
        import jax.numpy as jnp

        concatenated = jnp.concatenate(list(layer_acts.values()), axis=-1)
        layer_acts = {"concatenated": concatenated}
    elif layer_selection != "individual":
        raise ValueError(f"Invalid layer_selection: {layer_selection}. Must be 'individual' or 'concatenated'.")

    return {
        "activations": layer_acts,
        "belief_states": belief_states,
        "weights": weights,
    }


class ActivationAnalysis(Protocol):
    """Protocol for activation analysis implementations."""

    _requires_belief_states: bool
    _token_selection: str
    _layer_selection: str
    _use_probs_as_weights: bool

    def analyze(
        self,
        activations: Mapping[str, jax.Array],
        weights: jax.Array,
        belief_states: jax.Array | None = None,
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Analyze activations and return scalar metrics and projections."""
        ...


class ActivationTracker:
    """Orchestrates multiple activation analyses with efficient preprocessing."""

    def __init__(self, analyses: Mapping[str, ActivationAnalysis]):
        """Initialize the tracker with named analyses."""
        self._analyses = analyses

    def analyze(
        self,
        inputs: jax.Array,
        beliefs: jax.Array,
        probs: jax.Array,
        activations: Mapping[str, jax.Array],
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Run all analyses and return namespaced results."""
        preprocessing_cache: dict[tuple[str, str, bool], PreparedActivations] = {}

        for analysis in self._analyses.values():
            config_key = (
                analysis._token_selection,
                analysis._layer_selection,
                analysis._use_probs_as_weights,
            )

            if config_key not in preprocessing_cache:
                prepared = prepare_activations(
                    inputs=inputs,
                    beliefs=beliefs,
                    probs=probs,
                    activations=activations,
                    token_selection=analysis._token_selection,
                    layer_selection=analysis._layer_selection,
                    use_probs_as_weights=analysis._use_probs_as_weights,
                )
                preprocessing_cache[config_key] = prepared

        all_scalars = {}
        all_projections = {}

        for analysis_name, analysis in self._analyses.items():
            config_key = (
                analysis._token_selection,
                analysis._layer_selection,
                analysis._use_probs_as_weights,
            )
            prepared = preprocessing_cache[config_key]

            prepared_activations: Mapping[str, jax.Array] = prepared["activations"]  # type: ignore[assignment]
            prepared_beliefs = prepared["belief_states"]
            prepared_weights = prepared["weights"]

            if analysis._requires_belief_states and prepared_beliefs is None:
                raise ValueError(
                    f"Analysis '{analysis_name}' requires belief_states but none available after preprocessing. "
                    f"This should not happen - please report this as a bug."
                )

            scalars, projections = analysis.analyze(
                activations=prepared_activations,
                weights=prepared_weights,  # type: ignore[arg-type]
                belief_states=prepared_beliefs,
            )

            for key, value in scalars.items():
                all_scalars[f"{analysis_name}/{key}"] = value

            for key, value in projections.items():
                all_projections[f"{analysis_name}/{key}"] = value

        return all_scalars, all_projections
