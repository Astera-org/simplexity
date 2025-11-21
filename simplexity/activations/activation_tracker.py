"""Activation analysis for Transformer layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.typing import DTypeLike

from simplexity.activations.activation_analyses import ActivationAnalysis
from simplexity.utils.analysis_utils import build_deduplicated_dataset
from simplexity.utils.pytorch_utils import torch_to_jax


@dataclass
class PreparedActivations:
    """Prepared activations with belief states and sample weights."""

    activations: Mapping[str, jax.Array]
    belief_states: jax.Array | None
    weights: jax.Array


def _get_uniform_weights(n_samples: int, dtype: DTypeLike) -> jax.Array:
    """Get uniform weights that sum to 1."""
    weights = jnp.ones(n_samples, dtype=dtype)
    weights = weights / weights.sum()
    return weights


def _to_jax_array(value: Any) -> jax.Array:
    """Convert supported tensor types to JAX arrays."""
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, torch.Tensor):
        return torch_to_jax(value)
    return jnp.asarray(value)


def prepare_activations(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations: Mapping[str, jax.Array],
    last_token_only: bool = False,
    concat_layers: bool = False,
    use_probs_as_weights: bool = True,
) -> PreparedActivations:
    """Preprocess activations by deduplicating sequences, selecting tokens/layers, and computing weights."""
    inputs = _to_jax_array(inputs)
    beliefs = _to_jax_array(beliefs)
    probs = _to_jax_array(probs)
    activations = {name: _to_jax_array(layer) for name, layer in activations.items()}

    if last_token_only:
        beliefs = beliefs[:, -1, :]
        probs = probs[:, -1]
        activations = {name: acts[:, -1, :] for name, acts in activations.items()}

    dataset = build_deduplicated_dataset(
        inputs=inputs,
        beliefs=beliefs,
        probs=probs,
        activations_by_layer=activations,
        select_last_token=last_token_only,
    )

    layer_acts = dataset.activations_by_layer
    belief_states = dataset.beliefs
    weights = (
        dataset.probs if use_probs_as_weights else _get_uniform_weights(belief_states.shape[0], belief_states.dtype)
    )

    if concat_layers:
        concatenated = jnp.concatenate(list(layer_acts.values()), axis=-1)
        layer_acts = {"concatenated": concatenated}

    return PreparedActivations(
        activations=layer_acts,
        belief_states=belief_states,
        weights=weights,
    )


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
        preprocessing_cache: dict[tuple[bool, bool, bool], PreparedActivations] = {}

        for analysis in self._analyses.values():
            config_key = (
                analysis.last_token_only,
                analysis.concat_layers,
                analysis.use_probs_as_weights,
            )

            if config_key not in preprocessing_cache:
                prepared = prepare_activations(
                    inputs=inputs,
                    beliefs=beliefs,
                    probs=probs,
                    activations=activations,
                    last_token_only=analysis.last_token_only,
                    concat_layers=analysis.concat_layers,
                    use_probs_as_weights=analysis.use_probs_as_weights,
                )
                preprocessing_cache[config_key] = prepared

        all_scalars = {}
        all_projections = {}

        for analysis_name, analysis in self._analyses.items():
            config_key = (
                analysis.last_token_only,
                analysis.concat_layers,
                analysis.use_probs_as_weights,
            )
            prepared = preprocessing_cache[config_key]

            prepared_activations: Mapping[str, jax.Array] = prepared["activations"]
            prepared_beliefs = prepared["belief_states"]
            prepared_weights = prepared["weights"]

            if analysis.requires_belief_states and prepared_beliefs is None:
                raise ValueError(
                    f"Analysis '{analysis_name}' requires belief_states but none available after preprocessing."
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
