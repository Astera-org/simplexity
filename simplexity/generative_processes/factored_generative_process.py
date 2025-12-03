"""Unified factored generative process with pluggable conditional structures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.structures import ConditionalContext, ConditionalStructure
from simplexity.logger import SIMPLEXITY_LOGGER
from simplexity.utils.factoring_utils import TokenEncoder, transition_with_obs
from simplexity.utils.jnp_utils import resolve_jax_device

ComponentType = Literal["hmm", "ghmm"]
FactoredState = tuple[jnp.ndarray, ...]


def _move_arrays_to_device(
    arrays: Sequence[jnp.ndarray],
    device: jax.Device,  # type: ignore[valid-type]
    name: str,
) -> tuple[jnp.ndarray, ...]:
    """Move arrays to specified device with warning if needed.

    Args:
        arrays: Sequence of arrays to move
        device: Target device
        name: Name for warning messages (e.g., "Transition matrices")

    Returns:
        Tuple of arrays on target device
    """
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
    """Unified factored generative process with pluggable conditional structures.

    This class provides a single implementation of factored generative processes
    that supports different conditional dependency patterns via the ConditionalStructure protocol.

    Attributes:
        component_types: Type of each factor ("hmm" or "ghmm")
        transition_matrices: Per-factor transition tensors (shape [K_i, V_i, S_i, S_i])
        normalizing_eigenvectors: Per-factor eigenvectors (shape [K_i, S_i])
        initial_states: Initial state per factor (shape [S_i])
        num_variants: Number of parameter variants per factor
        structure: Conditional structure determining factor interactions
        encoder: Token encoder for composite observations
    """

    # Static structure
    component_types: tuple[ComponentType, ...]
    num_variants: tuple[int, ...]
    device: jax.Device  # type: ignore[valid-type]

    # Per-factor parameters
    transition_matrices: tuple[jnp.ndarray, ...]
    normalizing_eigenvectors: tuple[jnp.ndarray, ...]
    initial_states: tuple[jnp.ndarray, ...]

    # Conditional structure and encoding
    structure: ConditionalStructure
    encoder: TokenEncoder

    def __init__(
        self,
        *,
        component_types: Sequence[ComponentType],
        transition_matrices: Sequence[jnp.ndarray],
        normalizing_eigenvectors: Sequence[jnp.ndarray],
        initial_states: Sequence[jnp.ndarray],
        structure: ConditionalStructure,
        device: str | None = None,
    ) -> None:
        """Initialize factored generative process.

        Args:
            component_types: Type of each factor ("hmm" or "ghmm")
            transition_matrices: Per-factor transition tensors.
                transition_matrices[i] has shape [K_i, V_i, S_i, S_i]
            normalizing_eigenvectors: Per-factor eigenvectors for GHMM.
                normalizing_eigenvectors[i] has shape [K_i, S_i]
            initial_states: Initial state per factor (shape [S_i])
            structure: Conditional structure defining factor interactions
            device: Device to place arrays on (e.g., "cpu", "gpu")
        """
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
        for i, T in enumerate(self.transition_matrices):
            if T.ndim != 4:
                raise ValueError(f"transition_matrices[{i}] must have shape [K, V, S, S], got {T.shape}")
            K, V, S1, S2 = T.shape
            if S1 != S2:
                raise ValueError(f"transition_matrices[{i}] square mismatch: {S1} vs {S2}")
            vocab_sizes.append(V)
            num_variants.append(K)

        self.num_variants = tuple(int(k) for k in num_variants)
        self.encoder = TokenEncoder(jnp.array(vocab_sizes))

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

    # ------------------------ GenerativeProcess API -------------------------
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of composite observations."""
        return self.encoder.composite_vocab_size

    @property
    def initial_state(self) -> FactoredState:
        """Initial state across all factors."""
        return tuple(self.initial_states)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: FactoredState) -> jnp.ndarray:
        """Compute P(composite_token | state) under the conditional structure.

        Args:
            state: Tuple of state vectors (one per factor)

        Returns:
            Distribution over composite tokens, shape [prod(V_i)]
        """
        context = self._make_context(state)
        return self.structure.compute_joint_distribution(context)

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: FactoredState) -> jnp.ndarray:
        """Compute log P(composite_token | state).

        Args:
            log_belief_state: Tuple of log-state vectors

        Returns:
            Log-distribution over composite tokens, shape [prod(V_i)]
        """
        state = tuple(jnp.exp(s) for s in log_belief_state)
        probs = self.observation_probability_distribution(state)
        return jnp.log(probs)

    @eqx.filter_jit
    def emit_observation(self, state: FactoredState, key: jax.Array) -> jnp.ndarray:
        """Sample composite observation from current state.

        Args:
            state: Tuple of state vectors
            key: JAX random key

        Returns:
            Composite observation (scalar token)
        """
        probs = self.observation_probability_distribution(state)
        token_flat = jax.random.categorical(key, jnp.log(probs))
        return token_flat

    @eqx.filter_jit
    def transition_states(self, state: FactoredState, obs: chex.Array) -> FactoredState:
        """Update states given composite observation.

        Args:
            state: Tuple of current state vectors
            obs: Composite observation (scalar token)

        Returns:
            Tuple of updated state vectors
        """
        # Decode composite observation to per-factor tokens
        obs_tuple = self.encoder.token_to_tuple(obs)

        # Select variants based on conditional structure
        context = self._make_context(state)
        variants = self.structure.select_variants(obs_tuple, context)

        # Update each factor's state
        new_states: list[jnp.ndarray] = []
        for i, (s_i, t_i, k_i) in enumerate(zip(state, obs_tuple, variants, strict=True)):
            T_k = self.transition_matrices[i][k_i]
            norm_k = self.normalizing_eigenvectors[i][k_i] if self.component_types[i] == "ghmm" else None
            new_state_i = transition_with_obs(self.component_types[i], s_i, T_k, t_i, norm_k)
            new_states.append(new_state_i)

        return tuple(new_states)

    @eqx.filter_jit
    def probability(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Compute P(observations) by scanning through sequence.

        Args:
            observations: Array of composite observations

        Returns:
            Scalar probability
        """

        def step(carry: FactoredState, obs: jnp.ndarray):
            state = carry
            dist = self.observation_probability_distribution(state)
            p = dist[obs]
            new_state = self.transition_states(state, obs)
            return new_state, p

        _, ps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.prod(ps)

    @eqx.filter_jit
    def log_probability(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Compute log P(observations) by scanning through sequence.

        Args:
            observations: Array of composite observations

        Returns:
            Scalar log-probability
        """

        def step(carry: FactoredState, obs: jnp.ndarray):
            state = carry
            # Compute distribution directly without converting to log and back
            dist = self.observation_probability_distribution(state)
            lp = jnp.log(dist[obs])
            new_state = self.transition_states(state, obs)
            return new_state, lp

        _, lps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.sum(lps)
