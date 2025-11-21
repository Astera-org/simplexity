"""Generator for generative processes."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


@eqx.filter_jit
def generate_data_batch(
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    gen_states: jax.Array,
    data_generator: GenerativeProcess,
    batch_size: int,
    sequence_len: int,
    key: jax.Array,
    bos_token: int | None = None,
    eos_token: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generate sequences plus per-token belief states and prefix probabilities."""
    batch_keys = jax.random.split(key, batch_size)
    belief_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, True)
    next_states = belief_states[:, -1, :]

    prefix_probs = _compute_prefix_probabilities(data_generator, gen_states, tokens)

    if bos_token is not None:
        tokens = jnp.concatenate([jnp.full((batch_size, 1), bos_token), tokens], axis=1)
        prefix_probs = jnp.concatenate(
            [jnp.ones((batch_size, 1), dtype=prefix_probs.dtype), prefix_probs],
            axis=1,
        )
    if eos_token is not None:
        tokens = jnp.concatenate([tokens, jnp.full((batch_size, 1), eos_token)], axis=1)
        prefix_probs = jnp.concatenate(
            [prefix_probs, prefix_probs[:, -1:, ...]],
            axis=1,
        )

    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    prefix_probs = prefix_probs[:, : inputs.shape[1]]

    return next_states, belief_states, prefix_probs, inputs, labels


def _compute_prefix_probabilities(
    data_generator: GenerativeProcess,
    initial_states: jax.Array,
    tokens: jax.Array,
) -> jax.Array:
    def run_sequence(state: jax.Array, seq: jax.Array) -> jax.Array:
        def step(carry_state: jax.Array, token: jax.Array) -> tuple[jax.Array, jax.Array]:
            obs_probs = data_generator.observation_probability_distribution(carry_state)
            token_prob = obs_probs[token]
            new_state = data_generator.transition_states(carry_state, token)
            return new_state, token_prob

        _, token_probs = jax.lax.scan(step, state, seq)
        return jnp.cumprod(token_probs, axis=0)

    return jax.vmap(run_sequence)(initial_states, tokens)
