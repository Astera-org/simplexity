"""Activation analysis for Transformer layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.typing import DTypeLike

from simplexity.activations.activation_analyses import ActivationAnalysis
from simplexity.utils.analysis_utils import build_deduplicated_dataset
from simplexity.utils.pytorch_utils import torch_to_jax


@dataclass
class PreparedActivations:
    """Prepared activations with belief states and sample weights."""

    activations: Mapping[str, jax.Array]
    belief_states: jax.Array | tuple[jax.Array, ...] | None
    weights: jax.Array


class PrepareOptions(NamedTuple):
    """Configuration options for activation preparation."""

    last_token_only: bool
    concat_layers: bool
    use_probs_as_weights: bool


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


def _convert_tuple_to_jax_array(value: tuple[Any, ...]) -> tuple[jax.Array, ...]:
    """Convert a tuple of supported tensor types to JAX arrays."""
    return tuple(_to_jax_array(v) for v in value)


def prepare_activations(
    inputs: jax.Array | torch.Tensor | np.ndarray,
    beliefs: jax.Array
    | torch.Tensor
    | np.ndarray
    | tuple[jax.Array, ...]
    | tuple[torch.Tensor, ...]
    | tuple[np.ndarray, ...],
    probs: jax.Array | torch.Tensor | np.ndarray,
    activations: Mapping[str, jax.Array | torch.Tensor | np.ndarray],
    prepare_options: PrepareOptions,
) -> PreparedActivations:
    """Preprocess activations by deduplicating sequences, selecting tokens/layers, and computing weights."""
    inputs = _to_jax_array(inputs)
    beliefs = _convert_tuple_to_jax_array(beliefs) if isinstance(beliefs, tuple) else _to_jax_array(beliefs)
    probs = _to_jax_array(probs)
    activations = {name: _to_jax_array(layer) for name, layer in activations.items()}

    dataset = build_deduplicated_dataset(
        inputs=inputs,
        beliefs=beliefs,
        probs=probs,
        activations_by_layer=activations,
        select_last_token=prepare_options.last_token_only,
    )

    layer_acts = dataset.activations_by_layer
    belief_states = dataset.beliefs
    weights = (
        dataset.probs
        if prepare_options.use_probs_as_weights
        else _get_uniform_weights(dataset.probs.shape[0], dataset.probs.dtype)
    )

    if prepare_options.concat_layers:
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
        inputs: jax.Array | torch.Tensor | np.ndarray,
        beliefs: jax.Array | torch.Tensor | np.ndarray,
        probs: jax.Array | torch.Tensor | np.ndarray,
        activations: Mapping[str, jax.Array | torch.Tensor | np.ndarray],
    ) -> tuple[Mapping[str, float], Mapping[str, jax.Array]]:
        """Run all analyses and return namespaced results."""
        preprocessing_cache: dict[PrepareOptions, PreparedActivations] = {}

        for analysis in self._analyses.values():
            prepare_options = PrepareOptions(
                analysis.last_token_only,
                analysis.concat_layers,
                analysis.use_probs_as_weights,
            )
            config_key = prepare_options

            if config_key not in preprocessing_cache:
                prepared = prepare_activations(
                    inputs=inputs,
                    beliefs=beliefs,
                    probs=probs,
                    activations=activations,
                    prepare_options=prepare_options,
                )
                preprocessing_cache[config_key] = prepared

        all_scalars = {}
        all_projections = {}

        for analysis_name, analysis in self._analyses.items():
            prepare_options = PrepareOptions(
                analysis.last_token_only,
                analysis.concat_layers,
                analysis.use_probs_as_weights,
            )
            prepared = preprocessing_cache[prepare_options]

            prepared_activations: Mapping[str, jax.Array] = prepared.activations
            prepared_beliefs = prepared.belief_states
            prepared_weights = prepared.weights

            if analysis.requires_belief_states and prepared_beliefs is None:
                raise ValueError(
                    f"Analysis '{analysis_name}' requires belief_states but none available after preprocessing."
                )

            scalars, projections = analysis.analyze(
                activations=prepared_activations,
                weights=prepared_weights,
                belief_states=prepared_beliefs,
            )
            all_scalars.update({f"{analysis_name}/{key}": value for key, value in scalars.items()})
            all_projections.update({f"{analysis_name}/{key}": value for key, value in projections.items()})

        return all_scalars, all_projections
