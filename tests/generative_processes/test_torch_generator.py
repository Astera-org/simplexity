"""Test the torch generator module."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import jax
import jax.numpy as jnp
import torch

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)


def test_generate_data_batch():
    """Test generating a batch of data."""
    hmm = build_hidden_markov_model(process_name="zero_one_random", process_params={"p": 0.5})
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    gen_states, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key)
    assert isinstance(gen_states, jax.Array)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert inputs.shape == (batch_size, sequence_len - 1)
    assert labels.shape == (batch_size, sequence_len - 1)
    assert torch.all(inputs >= 0)
    assert torch.all(inputs < hmm.vocab_size)
    assert torch.all(labels >= 0)
    assert torch.all(labels < hmm.vocab_size)
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_bos_token():
    """Test generating a batch of data with a BOS token."""
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
    assert isinstance(gen_states, jax.Array)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert torch.all(inputs >= 0)
    assert torch.all(inputs[:, 0] == bos_token)
    assert torch.all(inputs[:, 1:] < bos_token)
    assert torch.all(labels >= 0)
    assert torch.all(labels < bos_token)
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_eos_token():
    """Test generating a batch of data with an EOS token."""
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
    assert isinstance(gen_states, jax.Array)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert inputs.shape == (batch_size, sequence_len)
    assert labels.shape == (batch_size, sequence_len)
    assert torch.all(inputs >= 0)
    assert torch.all(inputs < eos_token)
    assert torch.all(labels >= 0)
    assert torch.all(labels[:, :-1] < eos_token)
    assert torch.all(labels[:, -1] == eos_token)
    assert torch.equal(inputs[:, 1:], labels[:, :-1])
    assert gen_states.shape == (batch_size, *gen_state.shape)


def test_generate_data_batch_with_full_history():
    """Torch generator should surface belief states when requested."""
    hmm = build_hidden_markov_model("zero_one_random", process_params={"p": 0.5})
    batch_size = 3
    sequence_len = 5
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(123)
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

    assert isinstance(belief_states, jax.Array)
    assert isinstance(prefix_probs, jax.Array)
    assert isinstance(inputs, torch.Tensor)

    assert belief_states.shape == (batch_size, sequence_len, gen_state.shape[0])
    assert prefix_probs.shape == (batch_size, inputs.shape[1])
