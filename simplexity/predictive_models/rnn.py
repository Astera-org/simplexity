import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.predictive_models.predictive_model import PredictiveModel


class RNN(PredictiveModel):
    """A simple RNN model."""

    hidden_size: int = eqx.static_field()
    layers: eqx.nn.Sequential

    def __init__(self, in_size: int, out_size: int, hidden_size: int, *, key: chex.PRNGKey):
        cell_key, linear_key = jax.random.split(key)
        self.hidden_size = hidden_size
        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.GRUCell(in_size, hidden_size, key=cell_key),
                eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=linear_key),
            ]
        )

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the RNN."""
        hidden = jnp.zeros(self.hidden_size)

        def f(carry, x):
            return self.layers[0](x, carry), None

        out, _ = jax.lax.scan(f, hidden, xs)
        return self.layers[1](out)
