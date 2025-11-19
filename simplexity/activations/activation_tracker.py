"""Activation analysis for Transformer layers."""

from collections.abc import Mapping
from typing import Literal, TypedDict

import jax.numpy as jnp
from jax import Array as JaxArray
from jax.typing import DTypeLike

from simplexity.activations.activation_analyses import ActivationAnalysis
from simplexity.utils.analysis_utils import build_last_token_dataset, build_prefix_dataset


class PreparedActivations(TypedDict):
    """Prepared activations with belief states and sample weights."""

    activations: Mapping[str, JaxArray]
    belief_states: JaxArray | None
    weights: JaxArray


def _get_uniform_weights(n_samples: int, dtype: DTypeLike) -> JaxArray:
    """Get uniform weights that sum to 1."""
    weights = jnp.ones(n_samples, dtype=dtype)
    weights = weights / weights.sum()
    return weights


def prepare_activations(
    inputs: JaxArray,
    beliefs: JaxArray,
    probs: JaxArray,
    activations: Mapping[str, JaxArray],
    token_selection: Literal["all", "last"],
    concat_layers: bool = False,
    use_probs_as_weights: bool = True,
) -> PreparedActivations:
    """Preprocess activations by deduplicating sequences, selecting tokens/layers, and computing weights."""
    weights = None
    if token_selection == "all":
        prefix_dataset = build_prefix_dataset(inputs, beliefs, probs, dict(activations))
        belief_states = prefix_dataset.beliefs
        layer_acts = prefix_dataset.activations_by_layer

        if use_probs_as_weights:
            weights = prefix_dataset.probs

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
        raise ValueError(f"Invalid token_selection: {token_selection}. Must be 'all' or 'last'.")

    if weights is None:
        weights = _get_uniform_weights(belief_states.shape[0], dtype=beliefs.dtype)

    if concat_layers:
        concatenated = jnp.concatenate(list(layer_acts.values()), axis=-1)
        layer_acts = {"concatenated": concatenated}
    
    return {
        "activations": layer_acts,
        "belief_states": belief_states,
        "weights": weights,
    }


class ActivationTracker:
    """Orchestrates multiple activation analyses with efficient preprocessing."""

    def __init__(self, analyses: Mapping[str, ActivationAnalysis]):
        """Initialize the tracker with named analyses."""
        self._analyses = analyses

    def analyze(
        self,
        inputs: JaxArray,
        beliefs: JaxArray,
        probs: JaxArray,
        activations: Mapping[str, JaxArray],
    ) -> tuple[Mapping[str, float], Mapping[str, JaxArray]]:
        """Run all analyses and return namespaced results."""
        preprocessing_cache: dict[tuple[str, str, bool], PreparedActivations] = {}

        for analysis in self._analyses.values():
            config_key = (
                analysis._token_selection,
                str(analysis._concat_layers),
                analysis._use_probs_as_weights,
            )

            if config_key not in preprocessing_cache:
                prepared = prepare_activations(
                    inputs=inputs,
                    beliefs=beliefs,
                    probs=probs,
                    activations=activations,
                    token_selection=analysis._token_selection,
                    concat_layers=analysis._concat_layers,
                    use_probs_as_weights=analysis._use_probs_as_weights,
                )
                preprocessing_cache[config_key] = prepared

        all_scalars = {}
        all_projections = {}

        for analysis_name, analysis in self._analyses.items():
            config_key = (
                analysis._token_selection,
                str(analysis._concat_layers),
                analysis._use_probs_as_weights,
            )
            prepared = preprocessing_cache[config_key]

            prepared_activations: Mapping[str, JaxArray] = prepared["activations"]
            prepared_beliefs = prepared["belief_states"]
            prepared_weights = prepared["weights"]

            if analysis._requires_belief_states and prepared_beliefs is None:
                raise ValueError(
                    f"Analysis '{analysis_name}' requires belief_states but none available after preprocessing. "
                    f"This should not happen - please report this as a bug."
                )

            scalars, projections = analysis.analyze(
                activations=prepared_activations,
                weights=prepared_weights,
                belief_states=prepared_beliefs,
            )

            for key, value in scalars.items():
                all_scalars[f"{analysis_name}/{key}"] = value

            for key, value in projections.items():
                all_projections[f"{analysis_name}/{key}"] = value

        return all_scalars, all_projections
