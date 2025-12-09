"""Test the generator module."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)


def test_generate_data_batch():
    """Test the generate_data_batch function."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    gen_states, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key)
    assert inputs.shape == (batch_size, sequence_len - 1)
    assert labels.shape == (batch_size, sequence_len - 1)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs < hmm.vocab_size)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < hmm.vocab_size)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_bos_token():
    """Test the generate_data_batch function with a BOS token."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    bos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        bos_token=bos_token,
    )
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs[:, 0] == bos_token)
    assert jnp.all(inputs[:, 1:] < bos_token)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < bos_token)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_eos_token():
    """Test the generate_data_batch function with an EOS token."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    eos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        eos_token=eos_token,
    )
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert jnp.all(inputs >= 0)
    assert jnp.all(inputs < eos_token)
    assert jnp.all(labels >= 0)
    assert jnp.all(labels[:, :-1] < eos_token)
    assert jnp.all(labels[:, -1] == eos_token)
    chex.assert_trees_all_equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_full_history():
    """Ensure belief states and prefix probabilities can be returned."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 4
    sequence_len = 6
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    result = generate_data_batch_with_full_history(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
    )
    # Extract and type-check all fields
    belief_states = result["belief_states"]
    prefix_probs = result["prefix_probabilities"]
    inputs = result["inputs"]
    labels = result["labels"]

    assert isinstance(belief_states, jax.Array)
    assert isinstance(prefix_probs, jax.Array)
    assert isinstance(inputs, jax.Array)
    assert isinstance(labels, jax.Array)

    assert belief_states.shape == (batch_size, sequence_len, gen_state.shape[0])
    assert prefix_probs.shape == (batch_size, inputs.shape[1])
    assert labels.shape == inputs.shape
