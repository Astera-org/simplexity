from collections.abc import Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class GRUFn(eqx.Module):
    """Apply a GRU cell to each element of the input sequence."""

    cell: eqx.nn.GRUCell

    def __init__(self, in_size: int, hidden_size: int, key: chex.PRNGKey):
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)

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

    def __init__(self, in_size: int, out_size: int, key: chex.PRNGKey):
        self.linear = eqx.nn.Linear(in_size, out_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the linear model."""

        def process_element(_, x):
            out = self.linear(x)
            return None, out

        _, outs = jax.lax.scan(process_element, None, xs)
        return outs


class GRURNN(eqx.Module):
    """A GRU-based RNN model."""

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    layers: eqx.nn.Sequential

    def __init__(self, in_size: int, out_size: int, hidden_sizes: Sequence[int], *, key: chex.PRNGKey):
        self.in_size = in_size
        self.out_size = out_size

        num_gru_layers = len(hidden_sizes)
        linear_key, *cell_keys = jax.random.split(key, num_gru_layers + 1)

        layers = []
        for hidden_size, cell_key in zip(hidden_sizes, cell_keys, strict=True):
            gru_fn = GRUFn(in_size, hidden_size, cell_key)
            gru_layer = eqx.nn.Lambda(gru_fn)
            layers.append(gru_layer)
            in_size = hidden_size
        linear_fn = LinearFn(in_size, out_size, linear_key)
        linear_layer = eqx.nn.Lambda(linear_fn)
        layers.append(linear_layer)
        self.layers = eqx.nn.Sequential(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the GRU RNN."""
        return self.layers(x)


def build_gru_rnn(vocab_size: int, num_layers: int, hidden_size: int, seed: int) -> GRURNN:
    """Build a GRU RNN model."""
    hidden_sizes = [hidden_size] * num_layers
    key = jax.random.PRNGKey(seed)
    return GRURNN(vocab_size, vocab_size, hidden_sizes, key=key)
