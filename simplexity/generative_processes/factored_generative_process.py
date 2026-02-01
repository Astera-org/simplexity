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
        encoder: Token encoder for composite observations (all factors)
        observable_encoder: Token encoder for observable factors only
        hidden_factor_indices: Indices of hidden factors
        observable_factor_indices: Indices of observable factors
    """

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
    encoder: TokenEncoder  # Encodes all factors (full joint)
    observable_encoder: TokenEncoder  # Encodes only observable factors

    # Hidden factor configuration
    hidden_factor_indices: frozenset[int]
    observable_factor_indices: tuple[int, ...]
    _vocab_sizes_tuple: tuple[int, ...]  # For JIT-compatible reshaping

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
        hidden_factor_indices: frozenset[int] | None = None,
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
            noise_epsilon: Noisy channel epsilon value
            hidden_factor_indices: Indices of factors that are hidden (not observable).
                Hidden factors still evolve state and influence other factors via structures,
                but their tokens are marginalized out of the observable output.
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
        self.num_variants = tuple(int(k) for k in num_variants)
        self._vocab_sizes_tuple = tuple(int(v) for v in vocab_sizes)  # Store as Python ints for JIT
        self.encoder = TokenEncoder(jnp.array(vocab_sizes))

        # Set up hidden factor configuration
        self.hidden_factor_indices = hidden_factor_indices or frozenset()
        num_factors = len(component_types)
        self.observable_factor_indices = tuple(i for i in range(num_factors) if i not in self.hidden_factor_indices)

        # Create encoder for observable factors only
        if self.hidden_factor_indices:
            self.observable_encoder = TokenEncoder.create_for_subset(
                jnp.array(vocab_sizes), self.observable_factor_indices
            )
        else:
            self.observable_encoder = self.encoder

        # Store noise parameters - blur matrix is for observable vocab only
        self.noise_epsilon = noise_epsilon
        if noise_epsilon > 0.0:
            observable_vocab_sizes = tuple(vocab_sizes[i] for i in self.observable_factor_indices)
            self._blur_matrix = compute_joint_blur_matrix(observable_vocab_sizes, noise_epsilon)
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
            hidden_factor_indices=self.hidden_factor_indices,
        )

    # ------------------------ GenerativeProcess API -------------------------
    @property
    def vocab_size(self) -> int:
        """Observable vocabulary size (excluding hidden factors)."""
        return self.observable_encoder.composite_vocab_size

    @property
    def full_vocab_size(self) -> int:
        """Full vocabulary size including hidden factors."""
        return self.encoder.composite_vocab_size

    @property
    def initial_state(self) -> FactoredState:
        """Initial state across all factors."""
        return tuple(self.initial_states)

    def _marginalize_hidden_factors(self, joint_dist: jax.Array) -> jax.Array:
        """Marginalize out hidden factors from a joint distribution.

        Args:
            joint_dist: Distribution over full joint vocab, shape [prod(V_i)]

        Returns:
            Distribution over observable tokens, shape [prod(V_i for i observable)]
        """
        if not self.hidden_factor_indices:
            return joint_dist

        # Reshape to multi-dimensional array [V_0, V_1, ..., V_{F-1}]
        # Use _vocab_sizes_tuple (Python ints) to avoid JAX tracer issues during JIT
        joint_shaped = joint_dist.reshape(self._vocab_sizes_tuple)

        # Sum over hidden factor dimensions (in reverse order to preserve indices)
        hidden_axes = sorted(self.hidden_factor_indices, reverse=True)
        for axis in hidden_axes:
            joint_shaped = jnp.sum(joint_shaped, axis=axis)

        # Flatten back to 1D
        return joint_shaped.reshape(-1)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: FactoredState) -> jax.Array:
        """Compute P(observable_token | state) under the conditional structure.

        For processes with hidden factors, this marginalizes over hidden factor tokens.

        Args:
            state: Tuple of state vectors (one per factor)

        Returns:
            Distribution over observable composite tokens, shape [prod(V_i for i observable)]
        """
        context = self._make_context(state)
        joint_dist = self.structure.compute_joint_distribution(context)

        # Marginalize over hidden factors
        obs_dist = self._marginalize_hidden_factors(joint_dist)

        if self._blur_matrix is not None:
            obs_dist = self._blur_matrix @ obs_dist

        return obs_dist

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: FactoredState) -> jax.Array:
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
    def emit_observation(self, state: FactoredState, key: jax.Array) -> jax.Array:
        """Sample observable composite observation from current state.

        For processes with hidden factors, this samples from the full joint
        and projects to observable factors.

        Args:
            state: Tuple of state vectors
            key: JAX random key

        Returns:
            Observable composite observation (scalar token)
        """
        if not self.hidden_factor_indices:
            # No hidden factors - sample directly from observable distribution
            probs = self.observation_probability_distribution(state)
            token_flat = jax.random.categorical(key, jnp.log(probs))
            return token_flat

        # Sample from full joint distribution (before marginalization)
        context = self._make_context(state)
        joint_dist = self.structure.compute_joint_distribution(context)
        full_token = jax.random.categorical(key, jnp.log(joint_dist))

        # Convert to tuple and project to observable factors
        full_tuple = self.encoder.token_to_tuple(full_token)
        obs_tuple = self.encoder.project_tuple_to_subset(full_tuple, self.observable_factor_indices)
        return self.observable_encoder.tuple_to_token(obs_tuple)

    def _infer_hidden_factor_tokens(
        self, state: FactoredState, obs_tuple: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, ...]:
        """Infer most likely hidden factor tokens given observable tokens and state.

        Uses the joint distribution conditioned on observable tokens to find the
        most likely hidden factor configuration.

        Args:
            state: Current state vectors (all factors)
            obs_tuple: Observable factor tokens (indexed by observable_factor_indices)

        Returns:
            Full token tuple with inferred hidden factor tokens
        """
        context = self._make_context(state)
        joint_dist = self.structure.compute_joint_distribution(context)

        # Reshape to multi-dimensional [V_0, V_1, ..., V_{F-1}]
        # Use _vocab_sizes_tuple (Python ints) to avoid JAX tracer issues during JIT
        vocab_sizes = self._vocab_sizes_tuple
        joint_shaped = joint_dist.reshape(vocab_sizes)

        # Index into observable dimensions to get conditional distribution over hidden
        # Build indexing tuple: observable factors get their token, hidden get slice
        num_factors = len(vocab_sizes)
        obs_idx = 0
        index_tuple: list[jax.Array | slice] = []
        for i in range(num_factors):
            if i in self.hidden_factor_indices:
                index_tuple.append(slice(None))
            else:
                index_tuple.append(obs_tuple[obs_idx])
                obs_idx += 1

        # Get conditional distribution over hidden factors
        cond_dist = joint_shaped[tuple(index_tuple)]

        # Find argmax over the flattened hidden distribution
        cond_flat = cond_dist.reshape(-1)
        best_hidden_flat = jnp.argmax(cond_flat)

        # Convert flat hidden index to per-hidden-factor tokens
        hidden_indices = sorted(self.hidden_factor_indices)
        hidden_vocab_sizes = [vocab_sizes[i] for i in hidden_indices]

        # Decode the flat index to per-hidden-factor tokens
        hidden_tokens: list[jax.Array] = []
        remaining = best_hidden_flat
        for v in reversed(hidden_vocab_sizes):
            hidden_tokens.append(remaining % v)
            remaining = remaining // v
        hidden_tokens = list(reversed(hidden_tokens))

        # Reconstruct full token tuple
        full_tuple: list[jax.Array] = []
        obs_idx = 0
        hidden_idx = 0
        for i in range(num_factors):
            if i in self.hidden_factor_indices:
                full_tuple.append(hidden_tokens[hidden_idx])
                hidden_idx += 1
            else:
                full_tuple.append(obs_tuple[obs_idx])
                obs_idx += 1

        return tuple(full_tuple)

    @eqx.filter_jit
    def transition_states(self, state: FactoredState, obs: chex.Array) -> FactoredState:
        """Update states given observable composite observation.

        For processes with hidden factors, this infers the most likely hidden
        factor tokens before performing the transition.

        Args:
            state: Tuple of current state vectors
            obs: Observable composite observation (scalar token)

        Returns:
            Tuple of updated state vectors (all factors, including hidden)
        """
        if not self.hidden_factor_indices:
            # No hidden factors - standard transition
            obs_tuple = self.encoder.token_to_tuple(obs)
            context = self._make_context(state)
            variants = self.structure.select_variants(obs_tuple, context)

            new_states: list[jax.Array] = []
            for i, (s_i, t_i, k_i) in enumerate(zip(state, obs_tuple, variants, strict=True)):
                transition_matrix_k = self.transition_matrices[i][k_i]
                norm_k = self.normalizing_eigenvectors[i][k_i] if self.component_types[i] == "ghmm" else None
                new_state_i = transition_with_obs(self.component_types[i], s_i, transition_matrix_k, t_i, norm_k)
                new_states.append(new_state_i)

            return tuple(new_states)

        # Decode observable tokens
        obs_tuple = self.observable_encoder.token_to_tuple(obs)

        # Infer hidden factor tokens
        full_tuple = self._infer_hidden_factor_tokens(state, obs_tuple)

        # Select variants based on full token tuple (hidden factors can influence variants)
        context = self._make_context(state)
        variants = self.structure.select_variants(full_tuple, context)

        # Update all factors' states
        new_states = []
        for i, (s_i, t_i, k_i) in enumerate(zip(state, full_tuple, variants, strict=True)):
            transition_matrix_k = self.transition_matrices[i][k_i]
            norm_k = self.normalizing_eigenvectors[i][k_i] if self.component_types[i] == "ghmm" else None
            new_state_i = transition_with_obs(self.component_types[i], s_i, transition_matrix_k, t_i, norm_k)
            new_states.append(new_state_i)

        return tuple(new_states)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute P(observations) by scanning through sequence.

        Args:
            observations: Array of composite observations

        Returns:
            Scalar probability
        """

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
        """Compute log P(observations) by scanning through sequence.

        Args:
            observations: Array of composite observations

        Returns:
            Scalar log-probability
        """

        def step(carry: FactoredState, obs: jax.Array):
            state = carry
            # Compute distribution directly without converting to log and back
            dist = self.observation_probability_distribution(state)
            lp = jnp.log(dist[obs])
            new_state = self.transition_states(state, obs)
            return new_state, lp

        _, lps = jax.lax.scan(step, self.initial_state, observations)
        return jnp.sum(lps)
