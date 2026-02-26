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
from simplexity.generative_processes.nonergodic_generative_process import NonErgodicState


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
    """Generate sequences plus per-token belief states and prefix probabilities."""
    batch_keys = jax.random.split(key, batch_size)
    belief_states, tokens = data_generator.generate(gen_states, batch_keys, sequence_len, True)

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

    if bos_token is None:
        # Drop first belief state since it's the initial state before any token
        belief_states = _slice_belief_states(belief_states, slice(1, None))

    input_len = inputs.shape[1]
    belief_states = _slice_belief_states(belief_states, slice(None, input_len))

    result = {
        "belief_states": belief_states,
        "prefix_probabilities": prefix_probs,
        "inputs": inputs,
        "labels": labels,
    }

    return result


def _slice_belief_states(
    belief_states: jax.Array | tuple[jax.Array, ...] | NonErgodicState,
    seq_slice: slice,
) -> jax.Array | tuple[jax.Array, ...] | NonErgodicState:
    """Slice belief states along the sequence dimension (axis 1).

    Handles different state representations:
    - Plain array: slice directly
    - Tuple of arrays: slice each element
    - NonErgodicState: slice both component_beliefs and component_states
    """
    if isinstance(belief_states, NonErgodicState):
        return NonErgodicState(
            component_beliefs=belief_states.component_beliefs[:, seq_slice, ...],
            component_states=tuple(
                tuple(s[:, seq_slice, ...] for s in cs) if isinstance(cs, tuple) else cs[:, seq_slice, ...]
                for cs in belief_states.component_states
            ),
        )
    elif isinstance(belief_states, tuple):
        return tuple(b[:, seq_slice, ...] for b in belief_states)
    else:
        return belief_states[:, seq_slice, ...]


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
