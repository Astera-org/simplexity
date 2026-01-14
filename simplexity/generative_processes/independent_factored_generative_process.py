"""Independent factored generative process with per-factor sampling and frozen factors."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.factored_generative_process import (
    ComponentType,
    FactoredGenerativeProcess,
    FactoredState,
)
from simplexity.generative_processes.structures import ConditionalStructure
from simplexity.generative_processes.structures.independent import IndependentStructure
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class IndependentFactoredGenerativeProcess(FactoredGenerativeProcess):
    """Factored generative process with independent per-factor sampling and frozen factors.

    This variant samples emissions from each factor independently (not from the joint
    distribution), then combines them using TokenEncoder.tuple_to_token. It also supports
    "frozen" factors whose entire emission sequences are identical across batch samples.

    Frozen factors use keys derived from a stored `frozen_key`, while unfrozen factors
    use keys derived from the per-sample key. Since the same `frozen_key` produces the
    same derived keys across all batch samples, frozen factors naturally produce
    identical sequences.

    Attributes:
        frozen_factor_indices: frozenset of factor indices that are frozen
        frozen_key: JAX random key used for generating frozen sequences
    """

    frozen_factor_indices: jax.Array
    frozen_key: jax.Array

    def __init__(
        self,
        *,
        component_types: Sequence[ComponentType],
        transition_matrices: Sequence[jax.Array],
        normalizing_eigenvectors: Sequence[jax.Array],
        initial_states: Sequence[jax.Array],
        structure: ConditionalStructure,
        device: str | None = None,
        frozen_factor_indices: Sequence[int] = (),
        frozen_key: jax.Array | None = None,
    ) -> None:
        """Initialize independent factored generative process.

        Args:
            component_types: Type of each factor ("hmm" or "ghmm")
            transition_matrices: Per-factor transition tensors.
                transition_matrices[i] has shape [K_i, V_i, S_i, S_i]
            normalizing_eigenvectors: Per-factor eigenvectors for GHMM.
                normalizing_eigenvectors[i] has shape [K_i, S_i]
            initial_states: Initial state per factor (shape [S_i])
            structure: Conditional structure defining factor interactions
            device: Device to place arrays on (e.g., "cpu", "gpu")
            frozen_factor_indices: Indices of factors whose sequences are frozen across batch
            frozen_key: JAX random key for frozen sequence generation. Required if
                frozen_factor_indices is non-empty.

        Raises:
            ValueError: If frozen_factor_indices is non-empty but frozen_key is None
            ValueError: If frozen_factor_indices contains invalid indices
        """
        super().__init__(
            component_types=component_types,
            transition_matrices=transition_matrices,
            normalizing_eigenvectors=normalizing_eigenvectors,
            initial_states=initial_states,
            structure=structure,
            device=device,
        )

        num_factors = len(component_types)
        for idx in frozen_factor_indices:
            if idx < 0 or idx >= num_factors:
                raise ValueError(f"Invalid frozen factor index {idx}. Must be in [0, {num_factors})")

        if frozen_key is None:
            if frozen_factor_indices:
                raise ValueError("frozen_factor_indices must be empty if frozen_key is None")
            frozen_key = jax.random.PRNGKey(0)  # dummy value, will never be used
        self.frozen_factor_indices = jnp.isin(jnp.arange(num_factors), jnp.array(frozen_factor_indices))
        self.frozen_keys = jax.random.split(frozen_key, num_factors)

        if not isinstance(structure, IndependentStructure):
            SIMPLEXITY_LOGGER.warning(
                "IndependentFactoredGenerativeProcess is designed for IndependentStructure. "
                "Using %s may produce unexpected results.",
                type(structure).__name__,
            )

    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: jax.Array) -> jax.Array:
        """Sample composite observation by independently sampling each factor.

        Args:
            state: Tuple of state vectors (one per factor)
            key: JAX random key

        Returns:
            Composite observation (scalar token)
        """
        num_factors = len(self.component_types)

        factor_keys = jax.random.split(key, num_factors)
        factor_keys = jnp.where(self.frozen_factor_indices, self.frozen_keys, factor_keys)

        def get_per_factor_token(i: int, carry: jax.Array) -> jax.Array:
            T_i = self.transition_matrices[i][0]
            norm_i = self.normalizing_eigenvectors[i][0] if self.component_types[i] == "ghmm" else None
            p_i = compute_obs_dist_for_variant(self.component_types[i], state[i], T_i, norm_i)
            token_i = jax.random.categorical(factor_keys[i], jnp.log(p_i))
            return carry.at[i].set(token_i)

        per_factor_tokens = jax.lax.fori_loop(
            0, num_factors, get_per_factor_token, jnp.zeros(num_factors, dtype=jnp.int32)
        )

        return self.encoder.tuple_to_token(tuple(per_factor_tokens))
