"""Activation analysis for Transformer layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from jax.typing import DTypeLike
from omegaconf import DictConfig

from simplexity.activations.activation_analyses import ActivationAnalysis
from simplexity.activations.activation_visualizations import (
    ActivationVisualizationPayload,
    PreparedMetadata,
    build_visualization_payloads,
)
from simplexity.activations.visualization_persistence import save_visualization_payloads
from simplexity.activations.visualization_configs import build_activation_visualization_config
from simplexity.utils.analysis_utils import build_deduplicated_dataset
from simplexity.utils.pytorch_utils import torch_to_jax


@dataclass
class PreparedActivations:
    """Prepared activations with belief states and sample weights."""

    activations: Mapping[str, jax.Array]
    belief_states: jax.Array | None
    weights: jax.Array
    metadata: PreparedMetadata


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

    metadata = PreparedMetadata(
        sequences=dataset.sequences,
        steps=np.asarray([len(sequence) for sequence in dataset.sequences], dtype=np.int32),
        select_last_token=prepare_options.last_token_only,
    )

    return PreparedActivations(
        activations=layer_acts,
        belief_states=belief_states,
        weights=weights,
        metadata=metadata,
    )


class ActivationTracker:
    """Orchestrates multiple activation analyses with efficient preprocessing."""

    def __init__(
        self,
        analyses: Mapping[str, ActivationAnalysis],
        *,
        visualizations: Mapping[str, list[DictConfig | Mapping[str, Any]]] | None = None,
        default_backend: str = "altair",
    ):
        """Initialize the tracker with named analyses."""
        self._analyses = analyses
        self._default_backend = default_backend
        self._visualization_specs: dict[str, list[Any]] = {}
        self._scalar_history: dict[str, list[tuple[int, float]]] = {}
        if visualizations:
            for name, cfgs in visualizations.items():
                self._visualization_specs[name] = [build_activation_visualization_config(cfg) for cfg in cfgs]

    def analyze(
        self,
        inputs: jax.Array | torch.Tensor | np.ndarray,
        beliefs: jax.Array
        | torch.Tensor
        | np.ndarray
        | tuple[jax.Array, ...]
        | tuple[torch.Tensor, ...]
        | tuple[np.ndarray, ...],
        probs: jax.Array | torch.Tensor | np.ndarray,
        activations: Mapping[str, jax.Array | torch.Tensor | np.ndarray],
        step: int | None = None,
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
        all_visualizations: dict[str, ActivationVisualizationPayload] = {}

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

            namespaced_scalars = {f"{analysis_name}/{key}": value for key, value in scalars.items()}
            all_scalars.update(namespaced_scalars)
            all_projections.update({f"{analysis_name}/{key}": value for key, value in projections.items()})

            if step is not None:
                for scalar_key, scalar_value in namespaced_scalars.items():
                    if scalar_key not in self._scalar_history:
                        self._scalar_history[scalar_key] = []
                    self._scalar_history[scalar_key].append((step, float(scalar_value)))

            viz_configs = self._visualization_specs.get(analysis_name)
            if viz_configs:
                np_weights = np.asarray(prepared_weights)
                np_beliefs = None if prepared_beliefs is None else np.asarray(prepared_beliefs)
                np_projections = {key: np.asarray(value) for key, value in projections.items()}
                payloads = build_visualization_payloads(
                    analysis_name,
                    viz_configs,
                    default_backend=self._default_backend,
                    prepared_metadata=prepared.metadata,
                    weights=np_weights,
                    belief_states=np_beliefs,
                    projections=np_projections,
                    scalars={key: float(value) for key, value in scalars.items()},
                    scalar_history=self._scalar_history,
                    scalar_history_step=step,
                    analysis_concat_layers=analysis.concat_layers,
                    layer_names=list(prepared.activations.keys()),
                )
                all_visualizations.update(
                    {f"{analysis_name}/{payload.name}": payload for payload in payloads}
                )

        return all_scalars, all_projections, all_visualizations

    def save_visualizations(
        self,
        visualizations: Mapping[str, ActivationVisualizationPayload],
        root: Path,
        step: int,
    ) -> Mapping[str, str]:
        """Persist visualization payloads to disk with history accumulation."""

        return save_visualization_payloads(visualizations, root, step)

    def get_scalar_history(
        self,
        pattern: str | None = None,
    ) -> dict[str, list[tuple[int, float]]]:
        """Get scalar history, optionally filtered by pattern.

        Args:
            pattern: Optional wildcard pattern to filter scalar keys (e.g., "layer_*_rmse" or "layer_0...3_loss")

        Returns:
            Dictionary mapping scalar names to list of (step, value) tuples
        """
        if pattern is None:
            return dict(self._scalar_history)

        import re

        # Convert wildcard pattern to regex
        has_star = "*" in pattern
        has_range = bool(re.search(r"\d+\.\.\.\d+", pattern))

        if not has_star and not has_range:
            # No pattern, just exact match
            return {k: v for k, v in self._scalar_history.items() if k == pattern}

        # Expand range patterns first
        if has_range:
            range_match = re.search(r"(\d+)\.\.\.(\d+)", pattern)
            if range_match:
                start_idx = int(range_match.group(1))
                end_idx = int(range_match.group(2))
                patterns = []
                for idx in range(start_idx, end_idx):
                    patterns.append(re.sub(r"\d+\.\.\.\d+", str(idx), pattern))
            else:
                patterns = [pattern]
        else:
            patterns = [pattern]

        # Match against available keys
        matched = {}
        for p in patterns:
            if "*" in p:
                escaped = re.escape(p).replace(r"\*", r"([^/]+)")
                regex = re.compile(f"^{escaped}$")
                for key, history in self._scalar_history.items():
                    if regex.match(key):
                        matched[key] = history
            else:
                if p in self._scalar_history:
                    matched[p] = self._scalar_history[p]

        return matched

    def get_scalar_history_df(self) -> pd.DataFrame:
        """Export scalar history as a tidy pandas DataFrame.

        Returns:
            DataFrame with columns: metric, step, value
        """
        if not self._scalar_history:
            return pd.DataFrame(columns=["metric", "step", "value"])

        rows = []
        for metric_name, history in self._scalar_history.items():
            for step, value in history:
                rows.append({"metric": metric_name, "step": step, "value": value})

        return pd.DataFrame(rows)
