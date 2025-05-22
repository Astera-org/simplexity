from collections.abc import Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class EmbeddingFn(eqx.Module):
    """Apply an embedding model to each element of the input sequence."""

    embedding: eqx.nn.Embedding

    def __init__(self, vocab_size: int, embedding_size: int, key: chex.PRNGKey):
        self.embedding = eqx.nn.Embedding(vocab_size, embedding_size, key=key)

    def __call__(self, xs: jax.Array, **kwargs) -> jax.Array:
        """Forward pass of the linear model."""

        def process_element(_, x):
            out = self.embedding(x)
            return None, out

        _, outs = jax.lax.scan(process_element, None, xs)
        return outs


class GRUFn(eqx.Module):
    """Apply a GRU cell to each element of the input sequence."""

    cell: eqx.nn.GRUCell

    def __init__(self, input_size: int, hidden_size: int, key: chex.PRNGKey):
        self.cell = eqx.nn.GRUCell(input_size, hidden_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the GRU cell."""

        def process_element(carry, x):
            next_carry = self.cell(x, carry)
            return next_carry, next_carry

        hidden = jnp.zeros(self.cell.hidden_size)
        _, outs = jax.lax.scan(process_element, hidden, xs)
        return outs


class LinearFn(eqx.Module):
    """Apply a linear model to each element of the input sequence."""

    linear: eqx.nn.Linear

    def __init__(self, input_size: int, out_size: int, key: chex.PRNGKey):
        self.linear = eqx.nn.Linear(input_size, out_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the linear model."""

        def process_element(_, x):
            out = self.linear(x)
            return None, out

        _, outs = jax.lax.scan(process_element, None, xs)
        return outs


class GRURNN(eqx.Module):
    """A GRU-based RNN model."""

    vocab_size: int = eqx.field(static=True)
    layers: eqx.nn.Sequential

    def __init__(self, vocab_size: int, embedding_size: int, hidden_sizes: Sequence[int], *, key: chex.PRNGKey):
        self.vocab_size = vocab_size

        num_gru_layers = len(hidden_sizes)
        embedding_key, linear_key, *cell_keys = jax.random.split(key, num_gru_layers + 2)

        layers = []
        layers.append(EmbeddingFn(vocab_size, embedding_size, embedding_key))
        input_size = embedding_size
        for hidden_size, cell_key in zip(hidden_sizes, cell_keys, strict=True):
            gru_fn = GRUFn(input_size, hidden_size, cell_key)
            gru_layer = eqx.nn.Lambda(gru_fn)
            layers.append(gru_layer)
            input_size = hidden_size
        linear_fn = LinearFn(input_size, vocab_size, linear_key)
        linear_layer = eqx.nn.Lambda(linear_fn)
        layers.append(linear_layer)
        self.layers = eqx.nn.Sequential(layers)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the GRU RNN."""
        return self.layers(xs)


def build_gru_rnn(vocab_size: int, embedding_size: int, num_layers: int, hidden_size: int, seed: int) -> GRURNN:
    """Build a GRU RNN model."""
    hidden_sizes = [hidden_size] * num_layers
    key = jax.random.PRNGKey(seed)
    return GRURNN(vocab_size, embedding_size, hidden_sizes, key=key)
