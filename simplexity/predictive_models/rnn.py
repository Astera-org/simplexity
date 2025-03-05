import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.predictive_models.predictive_model import PredictiveModel


class RNN(PredictiveModel):
    """A simple RNN model."""

    hidden_size: int = eqx.static_field()
    cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear

    def __init__(self, in_size: int, out_size: int, hidden_size: int, *, key: chex.PRNGKey):
        cell_key, linear_key = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=cell_key)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=linear_key)

    @eqx.filter_jit
    def __call__(self, xs: jax.Array) -> jax.Array:
        """Forward pass of the RNN."""
        hidden = jnp.zeros(self.hidden_size)

        def f(carry, x):
            return self.cell(x, carry), None

        out, _ = jax.lax.scan(f, hidden, xs)
        return self.linear(out)
