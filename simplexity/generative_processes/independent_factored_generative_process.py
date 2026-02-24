"""Independent factored generative process with per-factor sampling and frozen factors."""

from __future__ import annotations

from collections.abc import Sequence

import chex
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

    frozen_factor_indices: frozenset[int]
    frozen_key: jax.Array | None

    def __init__(
        self,
        *,
        component_types: Sequence[ComponentType],
        transition_matrices: Sequence[jax.Array],
        normalizing_eigenvectors: Sequence[jax.Array],
        initial_states: Sequence[jax.Array],
        structure: ConditionalStructure,
        device: str | None = None,
        noise_epsilon: float = 0.0,
        frozen_factor_indices: frozenset[int] = frozenset(),
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
            noise_epsilon: Noisy channel epsilon value
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
            noise_epsilon=noise_epsilon,
        )

        num_factors = len(component_types)
        for idx in frozen_factor_indices:
            if idx < 0 or idx >= num_factors:
                raise ValueError(f"Invalid frozen factor index {idx}. Must be in [0, {num_factors})")

        if frozen_factor_indices and frozen_key is None:
            raise ValueError("frozen_key is required when frozen_factor_indices is non-empty")

        if not isinstance(structure, IndependentStructure):
            SIMPLEXITY_LOGGER.warning(
                "IndependentFactoredGenerativeProcess is designed for IndependentStructure. "
                "Using %s may produce unexpected results.",
                type(structure).__name__,
            )

        self.frozen_factor_indices = frozen_factor_indices
        self.frozen_key = frozen_key

    def _emit_observation_per_factor(self, state: FactoredState, key: jax.Array, frozen_key: jax.Array) -> jax.Array:
        """Sample each factor independently, choosing key based on frozen status.

        Args:
            state: Tuple of state vectors (one per factor)
            key: JAX random key for unfrozen factors
            frozen_key: JAX random key for frozen factors

        Returns:
            Composite observation (scalar token)
        """
        num_factors = len(self.component_types)

        factor_keys = jax.random.split(key, num_factors)
        frozen_factor_keys = jax.random.split(frozen_key, num_factors)

        per_factor_tokens = []
        for i in range(num_factors):
            if i in self.frozen_factor_indices:
                factor_key = frozen_factor_keys[i]
            else:
                factor_key = factor_keys[i]

            T_i = self.transition_matrices[i][0]
            norm_i = self.normalizing_eigenvectors[i][0] if self.component_types[i] == "ghmm" else None
            p_i = compute_obs_dist_for_variant(self.component_types[i], state[i], T_i, norm_i)

            token_i = jax.random.categorical(factor_key, jnp.log(p_i))
            per_factor_tokens.append(token_i)

        return self.encoder.tuple_to_token(tuple(per_factor_tokens))

    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: jax.Array) -> jax.Array:
        """Sample composite observation by independently sampling each factor.

        Args:
            state: Tuple of state vectors (one per factor)
            key: JAX random key

        Returns:
            Composite observation (scalar token)
        """
        frozen_key = self.frozen_key if self.frozen_key is not None else key
        return self._emit_observation_per_factor(state, key, frozen_key)

    @eqx.filter_vmap(in_axes=(None, 0, 0, None, None))
    def generate(
        self, state: FactoredState, key: chex.PRNGKey, sequence_len: int, return_all_states: bool
    ) -> tuple[FactoredState, chex.Array]:
        """Generate sequences with frozen factor support.

        For frozen factors, the same key stream is used across all batch samples,
        producing identical emission sequences. For unfrozen factors, each batch
        sample uses its own key stream, producing varying sequences.

        Args:
            state: Initial states, one per factor
            key: Random key for this batch sample
            sequence_len: Number of timesteps to generate
            return_all_states: Whether to return all intermediate states

        Returns:
            Tuple of (final_states or all_states, observations)
        """
        keys = jax.random.split(key, sequence_len)
        frozen_keys = jax.random.split(self.frozen_key, sequence_len) if self.frozen_key is not None else keys

        def gen_obs(
            carry_state: FactoredState, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[FactoredState, chex.Array]:
            key_t, frozen_key_t = inputs
            obs = self._emit_observation_per_factor(carry_state, key_t, frozen_key_t)
            new_state = self.transition_states(carry_state, obs)
            return new_state, obs

        def gen_states_and_obs(
            carry_state: FactoredState, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[FactoredState, tuple[FactoredState, chex.Array]]:
            key_t, frozen_key_t = inputs
            obs = self._emit_observation_per_factor(carry_state, key_t, frozen_key_t)
            new_state = self.transition_states(carry_state, obs)
            return new_state, (carry_state, obs)

        if return_all_states:
            _, (states, obs) = jax.lax.scan(gen_states_and_obs, state, (keys, frozen_keys))
            return states, obs

        return jax.lax.scan(gen_obs, state, (keys, frozen_keys))
