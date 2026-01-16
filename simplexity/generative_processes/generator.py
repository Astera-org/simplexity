"""Generator for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array | tuple[jax.Array, ...],
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[jax.Array | tuple[jax.Array, ...], jax.Array, jax.Array]:
    """Generate a batch of data without tracking intermediate beliefs."""
    batch_keys = jax.random.split(key, batch_size)
    gen_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, False)

    if bos_token is not None:
        tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), tokens], axis=1)
    if eos_token is not None:
        tokens = jnp.concatenate([tokens, jnp.full((batch_size, 1), eos_token)], axis=1)

    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    return gen_states, inputs, labels


@eqx.filter_jit
def generate_data_batch_with_full_history(
    gen_states: jax.Array | tuple[jax.Array, ...],
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> dict[str, jax.Array | tuple[jax.Array, ...]]:
    """Generate sequences plus per-token belief states, prefix probabilities, and observation log probs."""
    batch_keys = jax.random.split(key, batch_size)
    belief_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, True)

    prefix_probs = _compute_prefix_probabilities(data_generator, gen_states, tokens)

    # Always compute joint observation log probs
    observation_log_probs = _compute_observation_log_probs(data_generator, gen_states, tokens)

    # Check if this is a factored process with per-factor distributions
    is_factored = hasattr(data_generator, "factor_observation_probability_distributions")
    factor_observation_log_probs: tuple[jax.Array, ...] | None = None
    if is_factored:
        factor_observation_log_probs = _compute_factor_observation_log_probs(
            data_generator, gen_states, tokens  # type: ignore[arg-type]
        )

    if bos_token is not None:
        tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), tokens], axis=1)
        prefix_probs = jnp.concatenate(
            [jnp.ones((batch_size, 1), dtype=prefix_probs.dtype), prefix_probs],
            axis=1,
        )
        # Pad with zeros (log(1) = 0 for uniform, though this position is typically skipped)
        observation_log_probs = jnp.concatenate(
            [jnp.zeros((batch_size, 1, observation_log_probs.shape[-1]), dtype=observation_log_probs.dtype), observation_log_probs],
            axis=1,
        )
        if factor_observation_log_probs is not None:
            factor_observation_log_probs = tuple(
                jnp.concatenate([jnp.zeros((batch_size, 1, lp.shape[-1]), dtype=lp.dtype), lp], axis=1)
                for lp in factor_observation_log_probs
            )
    if eos_token is not None:
        tokens = jnp.concatenate([tokens, jnp.full((batch_size, 1), eos_token)], axis=1)
        prefix_probs = jnp.concatenate(
            [prefix_probs, prefix_probs[:, -1:, ...]],
            axis=1,
        )
        observation_log_probs = jnp.concatenate(
            [observation_log_probs, observation_log_probs[:, -1:, ...]],
            axis=1,
        )
        if factor_observation_log_probs is not None:
            factor_observation_log_probs = tuple(
                jnp.concatenate([lp, lp[:, -1:, ...]], axis=1) for lp in factor_observation_log_probs
            )

    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    prefix_probs = prefix_probs[:, : inputs.shape[1]]
    observation_log_probs = observation_log_probs[:, : inputs.shape[1]]
    if factor_observation_log_probs is not None:
        factor_observation_log_probs = tuple(lp[:, : inputs.shape[1]] for lp in factor_observation_log_probs)

    if bos_token is None:
        # Drop first belief state since it's the initial state before any token
        if isinstance(belief_states, tuple):
            belief_states = tuple(b[:, 1:, ...] for b in belief_states)
        else:
            belief_states = belief_states[:, 1:, ...]

    input_len = inputs.shape[1]
    if isinstance(belief_states, tuple):
        belief_states = tuple(b[:, :input_len, ...] for b in belief_states)
    else:
        belief_states = belief_states[:, :input_len, ...]

    result: dict[str, jax.Array | tuple[jax.Array, ...]] = {
        "belief_states": belief_states,
        "prefix_probabilities": prefix_probs,
        "observation_log_probs": observation_log_probs,
        "inputs": inputs,
        "labels": labels,
    }

    if factor_observation_log_probs is not None:
        result["factor_observation_log_probs"] = factor_observation_log_probs

    return result


def _compute_prefix_probabilities(
    data_generator: GenerativeProcess,
    initial_states: jax.Array | tuple[jax.Array, ...],
    tokens: jax.Array,
) -> jax.Array:
    def run_sequence(state: jax.Array | tuple[jax.Array, ...], seq: jax.Array) -> jax.Array:
        def step(carry_state: Any, token: jax.Array) -> tuple[Any, jax.Array]:
            obs_probs = data_generator.observation_probability_distribution(carry_state)
            token_prob = obs_probs[token]
            new_state = data_generator.transition_states(carry_state, token)
            return new_state, token_prob

        _, token_probs = jax.lax.scan(step, state, seq)
        return jnp.cumprod(token_probs, axis=0)

    return jax.vmap(run_sequence)(initial_states, tokens)


_LOG_PROB_EPS = 1e-10


def _compute_observation_log_probs(
    data_generator: GenerativeProcess,
    initial_states: jax.Array | tuple[jax.Array, ...],
    tokens: jax.Array,
) -> jax.Array:
    """Compute the full predictive log probability distribution at each position.

    Args:
        data_generator: The generative process used to compute observation probabilities.
        initial_states: Initial states for each sequence in the batch.
        tokens: Token sequences of shape (batch, seq_len).

    Returns:
        Log probability distributions of shape (batch, seq_len, vocab_size) where each
        position contains the full predictive distribution log P(X | state).
    """

    def run_sequence(state: jax.Array | tuple[jax.Array, ...], seq: jax.Array) -> jax.Array:
        def step(carry_state: Any, token: jax.Array) -> tuple[Any, jax.Array]:
            obs_probs = data_generator.observation_probability_distribution(carry_state)
            log_probs = jnp.log(obs_probs + _LOG_PROB_EPS)
            new_state = data_generator.transition_states(carry_state, token)
            return new_state, log_probs

        _, log_probs = jax.lax.scan(step, state, seq)
        return log_probs

    return jax.vmap(run_sequence)(initial_states, tokens)


def _compute_factor_observation_log_probs(
    data_generator: Any,
    initial_states: tuple[jax.Array, ...],
    tokens: jax.Array,
) -> tuple[jax.Array, ...]:
    """Compute per-factor predictive log probability distributions at each position.

    Args:
        data_generator: A FactoredGenerativeProcess with factor_observation_probability_distributions method.
        initial_states: Tuple of initial states for each factor in the batch.
        tokens: Token sequences of shape (batch, seq_len).

    Returns:
        Tuple of log probability distributions, one per factor. Each has shape
        (batch, seq_len, vocab_size_i) containing log P(X_i | state_i).
    """

    def run_sequence(
        state: tuple[jax.Array, ...], seq: jax.Array
    ) -> tuple[jax.Array, ...]:
        def step(
            carry_state: tuple[jax.Array, ...], token: jax.Array
        ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
            factor_probs = data_generator.factor_observation_probability_distributions(carry_state)
            factor_log_probs = tuple(jnp.log(p + _LOG_PROB_EPS) for p in factor_probs)
            new_state = data_generator.transition_states(carry_state, token)
            return new_state, factor_log_probs

        _, factor_log_probs_seq = jax.lax.scan(step, state, seq)
        return factor_log_probs_seq

    # vmap over the batch dimension
    batched_results = jax.vmap(run_sequence)(initial_states, tokens)
    # batched_results is a tuple of arrays, each of shape (batch, seq_len, vocab_size_i)
    return batched_results
