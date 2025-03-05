import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.predictive_models.predictive_model import PredictiveModel


class GRUFn(eqx.Module):
    """A simple RNN model."""

    hidden_size: int = eqx.static_field()
    cell: eqx.nn.GRUCell

    def __init__(self, in_size: int, hidden_size: int, key: chex.PRNGKey):
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the RNN."""

        def process_element(carry, x):
            return self.cell(x, carry), None

        hidden = jnp.zeros(self.hidden_size)
        out, _ = jax.lax.scan(process_element, hidden, xs)
        return out


class RNN(PredictiveModel):
    """A simple RNN model."""

    layers: eqx.nn.Sequential

    def __init__(self, in_size: int, out_size: int, hidden_size: int, *, key: chex.PRNGKey):
        cell_key, linear_key = jax.random.split(key)

        gru_fn = GRUFn(in_size, hidden_size, key=cell_key)

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(gru_fn),
                eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=linear_key),
            ]
        )

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the RNN."""
        return self.layers(xs)
