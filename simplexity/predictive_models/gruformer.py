from collections.abc import Mapping

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer.model_parts import TransformerLM
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer
from penzai.nn.layer import Layer as PenzaiModel

# Type alias for Linear or LinearInPlace
_LinearType = pz.nn.Linear | pz.nn.LinearInPlace  # type: ignore


class RNNBlock(eqx.Module):
    """A simple RNN block."""

    project_from_embedding: _LinearType  # type: ignore
    gru_cell: eqx.nn.GRUCell
    project_to_embedding: _LinearType  # type: ignore

    def __init__(self, num_heads: int, embedding_dim: int, projection_dim: int, key: chex.PRNGKey, **kwargs):
        keys = jax.random.split(key, 3)
        self.project_from_embedding = pz.nn.Linear.from_config(  # type: ignore
            name="project_from_embedding",
            init_base_rng=keys[0],
            input_axes={"embedding": embedding_dim},
            output_axes={"heads": num_heads, "projection": projection_dim},
        )
        self.gru_cell = eqx.nn.GRUCell(projection_dim, projection_dim, key=keys[1])
        self.project_to_embedding = pz.nn.Linear.from_config(  # type: ignore
            name="project_to_embedding",
            init_base_rng=keys[2],
            input_axes={"heads": num_heads, "projection": projection_dim},
            output_axes={"embedding": embedding_dim},
        )

    def __call__(self, x: pz.nx.NamedArray, **side_inputs: Mapping[str, jax.Array]):  # type: ignore
        """Process the input sequence."""
        x = self.project_from_embedding(x)

        def step(hidden: jax.Array, x_pos: jax.Array) -> tuple[jax.Array, jax.Array]:
            over_batch = jax.vmap(self.gru_cell, in_axes=(0, 0), out_axes=0)
            over_heads = jax.vmap(over_batch, in_axes=(1, 1), out_axes=1)
            next_hidden = over_heads(x_pos, hidden)
            return next_hidden, next_hidden

        hidden = jnp.zeros(
            (x.named_axes["batch"], x.named_axes["heads"], x.named_axes["projection"]),
            dtype=x.data_array.dtype,
        )
        x_unwrapped = x.unwrap("seq", "batch", "heads", "projection")
        _, x_unwrapped = jax.lax.scan(step, hidden, x_unwrapped)
        x = pz.nx.wrap(x_unwrapped, "seq", "batch", "heads", "projection")

        x = self.project_to_embedding(x)
        return x


def attention_to_gru(attention: pz.nn.Attention) -> RNNBlock:  # type: ignore
    """Convert an attention layer to a RNN block."""
    linear: pz.nn.Linear = pz.select(attention.input_to_query).at_instances_of(pz.nn.Linear).pick_nth_selected(0).get()  # type: ignore
    kwargs: Mapping[str, int] = linear.weights.value.named_shape  # type: ignore
    key = jax.random.PRNGKey(0)
    data = jnp.mean(linear.weights.value.data_array)  # TODO: handle uninitialized weights
    key = jax.random.fold_in(key, data)
    return RNNBlock(
        num_heads=kwargs["head_groups"] * kwargs["query_heads"],
        embedding_dim=kwargs["embedding"],
        projection_dim=kwargs["projection"],
        key=key,
    )


def convert_transformer_to_gruformer(transformer: TransformerLM) -> PenzaiModel:
    """Convert a transformer to a RNN."""
    return pz.select(transformer).at_instances_of(pz.nn.Attention).apply(attention_to_gru)


def build_llamalike_gruformer(
    config: LlamalikeTransformerConfig, init_base_rng: chex.PRNGKey, name: str = "gruformer"
) -> PenzaiModel:
    """Build a GRUFormer model from a LlamalikeTransformerConfig."""
    # TODO: make init_base_rng an optional argument
    transformer = build_llamalike_transformer(config, init_base_rng=init_base_rng, name=name)
    return convert_transformer_to_gruformer(transformer)
