"""Tests for entropy rate computation."""

import jax.numpy as jnp

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.entropy_rate import (
    compute_entropy_rate,
    compute_entropy_rate_bits,
)


def test_entropy_rate_less_than_uniform():
    """Test that entropy rate is less than uniform distribution entropy."""
    hmm = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
    entropy = compute_entropy_rate(hmm, num_steps=5000)
    uniform_entropy = jnp.log(hmm.vocab_size)
    assert entropy < uniform_entropy


def test_entropy_rate_positive():
    """Test that entropy rate is positive."""
    hmm = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
    entropy = compute_entropy_rate(hmm, num_steps=5000)
    assert entropy > 0


def test_entropy_rate_bits_conversion():
    """Test that bits conversion is correct."""
    hmm = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
    entropy_nats = compute_entropy_rate(hmm, num_steps=5000, seed=123)
    entropy_bits = compute_entropy_rate_bits(hmm, num_steps=5000, seed=123)
    expected_bits = entropy_nats / float(jnp.log(2))
    assert jnp.isclose(entropy_bits, expected_bits, rtol=1e-5)


def test_entropy_rate_deterministic_with_seed():
    """Test that entropy rate is deterministic given the same seed."""
    hmm = build_hidden_markov_model("mess3", {"x": 0.15, "a": 0.6})
    entropy1 = compute_entropy_rate(hmm, num_steps=1000, seed=42)
    entropy2 = compute_entropy_rate(hmm, num_steps=1000, seed=42)
    assert entropy1 == entropy2
