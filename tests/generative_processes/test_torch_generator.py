import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.torch_generator import generate_data_batch

try:
    import torch
except ImportError as e:
    raise ImportError("To use PyTorch support install the torch extra:\nuv sync --extra pytorch") from e


def test_generate_data_batch():
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
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
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    bos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key, bos_token=bos_token)
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
    hmm = build_hidden_markov_model("zero_one_random", p=0.5)
    batch_size = 10
    sequence_len = 10
    gen_state: jax.Array = hmm.initial_state
    states = jnp.repeat(gen_state[None, :], batch_size, axis=0)
    key = jax.random.PRNGKey(0)
    eos_token = hmm.vocab_size
    gen_states, inputs, labels = generate_data_batch(states, hmm, batch_size, sequence_len, key, eos_token=eos_token)
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
