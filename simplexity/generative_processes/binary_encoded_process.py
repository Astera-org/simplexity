"""Binary-encoded generative process wrapper.

Wraps any GenerativeProcess by encoding its vocabulary into binary. Each base token
is encoded as a fixed-width big-endian binary string of ceil(log2(vocab_size)) bits.
The wrapper emits individual bits and only transitions the underlying state after
a complete binary word has been emitted.

Token encoding: token t is emitted as bits b_0, ..., b_{n-1} where
b_i = (t >> (n - 1 - i)) & 1 (big-endian / MSB first).
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


class BinaryEncodedState(NamedTuple):
    """State for a binary-encoded generative process.

    Attributes:
        base_state: The wrapped process's belief state.
        bit_position: Index of the next bit to emit (0 to num_bits - 1).
        accumulated_prefix: Integer built from bits emitted so far in the current word.
    """

    base_state: Any
    bit_position: jax.Array
    accumulated_prefix: jax.Array


class BinaryEncodedProcess[State](GenerativeProcess[BinaryEncodedState]):
    """Wraps a GenerativeProcess by encoding its vocabulary into binary.

    For a base process with vocab size V:
    - num_bits = ceil(log2(V)) binary digits per base token
    - New vocab size is 2 (binary: 0 or 1)
    - Each base token is emitted as num_bits binary digits (big-endian, MSB first)
    - The underlying state only transitions after all num_bits bits are emitted
    - Unused binary codes (when V is not a power of 2) have probability 0

    This encoding preserves the causal state structure of the base process. The
    additional state variables (bit_position, accumulated_prefix) are deterministic
    given the history and add no genuine memory.

    Args:
        base_process: The generative process to wrap. Must have vocab_size >= 2.
    """

    base_process: GenerativeProcess[State]
    num_bits: int
    _token_indices: jax.Array

    def __init__(self, base_process: GenerativeProcess[State]) -> None:
        if base_process.vocab_size < 2:
            raise ValueError(f"base process vocab_size must be >= 2, got {base_process.vocab_size}")
        self.base_process = base_process
        self.num_bits = math.ceil(math.log2(base_process.vocab_size))
        self._token_indices = jnp.arange(2**self.num_bits)

    @property
    def vocab_size(self) -> int:
        """The number of observations: always 2 (binary)."""
        return 2

    @property
    def initial_state(self) -> BinaryEncodedState:
        """The initial state: base process initial state at bit position 0 with empty prefix."""
        return BinaryEncodedState(
            base_state=self.base_process.initial_state,
            bit_position=jnp.array(0, dtype=jnp.int32),
            accumulated_prefix=jnp.array(0, dtype=jnp.int32),
        )

    @eqx.filter_jit
    def emit_observation(self, state: BinaryEncodedState, key: chex.PRNGKey) -> chex.Array:
        """Emit the next binary digit by sampling from the conditional bit distribution."""
        dist = self.observation_probability_distribution(state)
        return jax.random.categorical(key, jnp.log(dist))

    @eqx.filter_jit
    def transition_states(self, state: BinaryEncodedState, obs: chex.Array) -> BinaryEncodedState:
        """Update state: accumulate the observed bit, transitioning the base state when the word is complete.

        When bit_position < num_bits - 1, only the prefix and position are updated.
        When bit_position == num_bits - 1, the complete token is decoded and the base
        state is transitioned, then position and prefix are reset to 0.
        """
        new_prefix = state.accumulated_prefix * 2 + obs
        new_position = state.bit_position + 1
        is_complete = new_position >= self.num_bits

        safe_token = jnp.clip(new_prefix, 0, self.base_process.vocab_size - 1)
        transitioned_base = self.base_process.transition_states(state.base_state, safe_token)

        final_base_state = jax.tree.map(
            lambda new, old: jnp.where(is_complete, new, old),
            transitioned_base,
            state.base_state,
        )
        final_position = jnp.where(is_complete, jnp.array(0, dtype=jnp.int32), new_position)
        final_prefix = jnp.where(is_complete, jnp.array(0, dtype=jnp.int32), new_prefix)

        return BinaryEncodedState(final_base_state, final_position, final_prefix)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: BinaryEncodedState) -> jax.Array:
        """Compute P(next_bit | base_state, accumulated_prefix).

        Returns a distribution over {0, 1} conditioned on the current base state
        and the bits already emitted for the current token.
        """
        base_dist = self.base_process.observation_probability_distribution(state.base_state)
        padded_dist = jnp.zeros(2**self.num_bits).at[: self.base_process.vocab_size].set(base_dist)

        shift_amount = self.num_bits - state.bit_position - 1
        shifted = jnp.right_shift(self._token_indices, shift_amount)

        p_0 = jnp.sum(padded_dist * (shifted == state.accumulated_prefix * 2))
        p_1 = jnp.sum(padded_dist * (shifted == state.accumulated_prefix * 2 + 1))

        total = p_0 + p_1
        return jnp.array([p_0 / total, p_1 / total])

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: BinaryEncodedState) -> jax.Array:
        """Compute log P(next_bit | log_base_state, accumulated_prefix).

        The base_state component of log_belief_state should be in log space.
        """
        base_log_dist = self.base_process.log_observation_probability_distribution(log_belief_state.base_state)
        padded_log_dist = jnp.full(2**self.num_bits, -jnp.inf).at[: self.base_process.vocab_size].set(base_log_dist)

        shift_amount = self.num_bits - log_belief_state.bit_position - 1
        shifted = jnp.right_shift(self._token_indices, shift_amount)

        mask_0 = jnp.where(shifted == log_belief_state.accumulated_prefix * 2, padded_log_dist, -jnp.inf)
        mask_1 = jnp.where(shifted == log_belief_state.accumulated_prefix * 2 + 1, padded_log_dist, -jnp.inf)

        log_p_0 = jax.nn.logsumexp(mask_0)
        log_p_1 = jax.nn.logsumexp(mask_1)

        log_total = jax.nn.logsumexp(jnp.array([log_p_0, log_p_1]))
        return jnp.array([log_p_0 - log_total, log_p_1 - log_total])

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of a binary sequence.

        Handles sequences of any length, including those that end mid-token.
        For complete token sequences, the result equals the base process probability
        of the decoded token sequence.
        """

        def _scan_fn(
            state: BinaryEncodedState, bit: jax.Array
        ) -> tuple[BinaryEncodedState, jax.Array]:
            dist = self.observation_probability_distribution(state)
            bit_prob = dist[bit]
            new_state = self.transition_states(state, bit)
            return new_state, bit_prob

        _, bit_probs = jax.lax.scan(_scan_fn, self.initial_state, observations)
        return jnp.prod(bit_probs)

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of a binary sequence.

        Handles sequences of any length, including those that end mid-token.
        For complete token sequences, the result equals the base process log probability
        of the decoded token sequence.
        """

        def _scan_fn(
            state: BinaryEncodedState, bit: jax.Array
        ) -> tuple[BinaryEncodedState, jax.Array]:
            dist = self.observation_probability_distribution(state)
            log_bit_prob = jnp.log(dist[bit])
            new_state = self.transition_states(state, bit)
            return new_state, log_bit_prob

        _, log_bit_probs = jax.lax.scan(_scan_fn, self.initial_state, observations)
        return jnp.sum(log_bit_probs)
