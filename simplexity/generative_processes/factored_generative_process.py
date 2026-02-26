"""Unified factored generative process with pluggable conditional structures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.noisy_channel import compute_joint_blur_matrix
from simplexity.generative_processes.structures import ConditionalContext, ConditionalStructure
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.utils.factoring_utils import TokenEncoder, transition_with_obs
from simplexity.utils.jnp_utils import resolve_jax_device

ComponentType = Literal["hmm", "ghmm"]
FactoredState = tuple[jax.Array, ...]


def _move_arrays_to_device(
    arrays: Sequence[jax.Array],
    device: jax.Device,  # type: ignore[valid-type]
    name: str,
) -> tuple[jax.Array, ...]:
    """Move arrays to specified device with warning if needed."""
    result = []
    for i, arr in enumerate(arrays):
        if arr.device != device:
            SIMPLEXITY_LOGGER.warning(
                "%s[%d] on device %s but model is on device %s. Moving to model device.",
                name,
                i,
                arr.device,
                device,
            )
            arr = jax.device_put(arr, device)
        result.append(arr)
    return tuple(result)


class FactoredGenerativeProcess(GenerativeProcess[FactoredState]):
    """Unified factored generative process with pluggable conditional structures."""

    # Static structure
    component_types: tuple[ComponentType, ...]
    num_variants: tuple[int, ...]
    device: jax.Device  # type: ignore[valid-type]

    # Per-factor parameters
    transition_matrices: tuple[jax.Array, ...]
    normalizing_eigenvectors: tuple[jax.Array, ...]
    initial_states: tuple[jax.Array, ...]

    # Conditional structure and encoding
    structure: ConditionalStructure
    encoder: TokenEncoder

    # Noise parameters
    noise_epsilon: float
    _blur_matrix: jax.Array | None

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
    ) -> None:
        if len(component_types) == 0:
            raise ValueError("Must provide at least one component")

        self.device = resolve_jax_device(device)
        self.component_types = tuple(component_types)

        # Move all arrays to device
        self.transition_matrices = _move_arrays_to_device(transition_matrices, self.device, "Transition matrices")
        self.normalizing_eigenvectors = _move_arrays_to_device(
            normalizing_eigenvectors, self.device, "Normalizing eigenvectors"
        )
        self.initial_states = _move_arrays_to_device(initial_states, self.device, "Initial states")

        self.structure = structure

        # Validate shapes and compute derived sizes
        vocab_sizes = []
        num_variants = []
        for i, transition_matrix in enumerate(self.transition_matrices):
            if transition_matrix.ndim != 4:
                raise ValueError(
                    f"transition_matrices[{i}] must have shape [K, V, S, S], got {transition_matrix.shape}"
                )
            num_var, vocab_size, state_dim1, state_dim2 = transition_matrix.shape
            if state_dim1 != state_dim2:
                raise ValueError(f"transition_matrices[{i}] square mismatch: {state_dim1} vs {state_dim2}")
            vocab_sizes.append(vocab_size)
            num_variants.append(num_var)
        self.num_variants = tuple(num_variants)
        self.encoder = TokenEncoder(jnp.array(vocab_sizes))

        # Store noise parameters
        self.noise_epsilon = noise_epsilon
        if noise_epsilon > 0.0:
            self._blur_matrix = compute_joint_blur_matrix(tuple(vocab_sizes), noise_epsilon)
        else:
            self._blur_matrix = None

    def _make_context(self, state: FactoredState) -> ConditionalContext:
        """Create conditional context for structure methods."""
        return ConditionalContext(
            states=state,
            component_types=self.component_types,
            transition_matrices=self.transition_matrices,
            normalizing_eigenvectors=self.normalizing_eigenvectors,
            vocab_sizes=self.encoder.vocab_sizes,
            num_variants=self.num_variants,
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of composite observations."""
        return self.encoder.composite_vocab_size

    @property
    def initial_state(self) -> FactoredState:
        """Initial state across all factors."""
        return tuple(self.initial_states)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: FactoredState) -> jax.Array:
        """Compute P(composite_token | state) under the conditional structure."""
        context = self._make_context(state)
        joint_dist = self.structure.compute_joint_distribution(context)

        if self._blur_matrix is not None:
            joint_dist = self._blur_matrix @ joint_dist

        return joint_dist

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: FactoredState) -> jax.Array:
        """Compute log P(composite_token | state)."""
        state = tuple(jnp.exp(s) for s in log_belief_state)
        probs = self.observation_probability_distribution(state)
        return jnp.log(probs)

    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: jax.Array) -> jax.Array:
        """Sample a composite observation from the current state."""
        probs = self.observation_probability_distribution(state)
        token_flat = jax.random.categorical(key, jnp.log(probs))
        return token_flat

    @eqx.filter_jit
    def transition_states(self, state: FactoredState, obs: chex.Array) -> FactoredState:
        """Update states given a composite observation."""
        # Decode composite observation to per-factor tokens
        obs_tuple = self.encoder.token_to_tuple(obs)

        # Select variants based on conditional structure
        context = self._make_context(state)
        variants = self.structure.select_variants(obs_tuple, context)

        # Update each factor's state
        new_states: list[jax.Array] = []
        for i, (s_i, t_i, k_i) in enumerate(zip(state, obs_tuple, variants, strict=True)):
            transition_matrix_k = self.transition_matrices[i][k_i]
            norm_k = self.normalizing_eigenvectors[i][k_i] if self.component_types[i] == "ghmm" else None
            new_state_i = transition_with_obs(self.component_types[i], s_i, transition_matrix_k, t_i, norm_k)
            new_states.append(new_state_i)

        return tuple(new_states)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute P(observations) by scanning through the sequence."""

        def step(carry: FactoredState, obs: jax.Array):
            state = carry
            dist = self.observation_probability_distribution(state)
            p = dist[obs]
            new_state = self.transition_states(state, obs)
            return new_state, p

        _, ps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.prod(ps)

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute log P(observations) by scanning through the sequence."""

        def step(carry: FactoredState, obs: jax.Array):
            state = carry
            # Compute distribution directly without converting to log and back
            dist = self.observation_probability_distribution(state)
            lp = jnp.log(dist[obs])
            new_state = self.transition_states(state, obs)
            return new_state, lp

        _, lps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.sum(lps)
