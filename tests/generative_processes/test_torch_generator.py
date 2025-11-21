"""Test the torch generator module."""

import jax
import jax.numpy as jnp
import torch

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch


def test_generate_data_batch():
    """Test generating a batch of data."""
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    gen_states, probs, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key)
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
    assert probs.shape == (batch_size,)


def test_generate_data_batch_with_bos_token():
    """Test generating a batch of data with a BOS token."""
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    bos_token = hmm.vocab_size
    gen_states, probs, inputs, labels = generate_data_batch(
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
    assert probs.shape == (batch_size,)


def test_generate_data_batch_with_eos_token():
    """Test generating a batch of data with an EOS token."""
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    eos_token = hmm.vocab_size
    gen_states, probs, inputs, labels = generate_data_batch(
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
    assert probs.shape == (batch_size,)


def test_generate_data_batch_return_all_states():
    """Torch generator should surface belief states when requested."""
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 3
    sequence_len = 5
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(123)
    belief_states, prefix_probs, inputs, labels = generate_data_batch(
        states,
        hmm,
        batch_size,
        sequence_len,
        key,
        return_all_states=True,
        return_prefix_probabilities=True,
    )
    assert belief_states.shape == (batch_size, sequence_len, gen_state.shape[0])
    assert prefix_probs.shape == (batch_size, inputs.shape[1])
