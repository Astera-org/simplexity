from typing import Any
import dataclasses
import jax
import jax.numpy as jnp
from .predictive_model import PredictiveModel
from penzai import pz
from penzai.models.transformer.variants import llamalike_common as llamalike_transformer


@pz.pytree_dataclass
class AttentionProfile(pz.nn.Layer):
    """Compute the average attention profile over the target positions relative to
    the query.  Only averages over query positions >= horizon.
    Insert this layer at the end of the pz.nn.Attention's query_key_to_attn
    layer to profile the aggregate attention patterns.
    """

    wrapped: pz.nn.Layer
    horizon: int
    target_profile: pz.StateVariable[jax.Array] = dataclasses.field(
        default_factory=lambda: pz.StateVariable(None),
        init=False,
    )

    def __call__(self, attn_logits: pz.nx.NamedArray, /, **side_inputs):
        attention_probs = self.wrapped(attn_logits, **side_inputs)
        check_axes = ("seq", "batch", "head_groups", "query_heads", "kv_seq")
        input_axes = tuple(attention_probs.named_axes.keys())
        if input_axes != check_axes:
            raise RuntimeError(
                f"Received input with named axes: ({', '.join(input_axes)}), but expected ({', '.join(check_axes)})"
            )
        attention_QK = attention_probs.untag("head_groups", "batch", "query_heads").mean()  # [seq, kv_seq]
        attention_QK = attention_QK.untag("seq", "kv_seq")
        self.target_profile.value = attention_profile(attention_QK, self.horizon)
        return attention_probs


def attention_profile(attention_QK: jax.Array, horizon: int) -> jax.Array:
    """Computes average attention profile over target (key, value) positions relative
    to from query position.
    Inputs:
      attention_QK: [Q, K]
    Returns:
      profile_H, where H is Q-horizon
    """
    Q, K = attention_QK.positional_shape
    assert horizon <= Q, f"{horizon=} must be less than query length {Q=}"
    assert Q == K, f"{Q=} must be equal to {K=}"
    seq_inds = jnp.arange(horizon, Q)[:, None]
    kv_inds = jnp.arange(horizon)[None, :] + seq_inds - horizon
    slc_HK = attention_QK[seq_inds, kv_inds]
    return slc_HK.mean(axis=0)  # [H]


def build_transformer(
    num_kv_heads,
    query_head_multiplier,
    embedding_dim,
    projection_dim,
    mlp_hidden_dim,
    num_decoder_blocks,
    vocab_size,
    mlp_variant,
    tie_embedder_and_logits,
    attention_profile_horizon,
    seed,
):
    config = llamalike_transformer.LlamalikeTransformerConfig(
        num_kv_heads=num_kv_heads,
        query_head_multiplier=num_kv_heads,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        num_decoder_blocks=num_decoder_blocks,
        vocab_size=vocab_size,
        mlp_variant=mlp_variant,
        tie_embedder_and_logits=tie_embedder_and_logits,
    )
    transformer = llamalike_transformer.build_llamalike_transformer(
        config, init_base_rng=jax.random.key(seed), name="llama_like"
    )
    if isinstance(attention_profile_horizon, int):
        transformer = (
            pz.select(transformer)
            .at_instances_of(pz.nn.Attention)
            .at(lambda x: x.query_key_to_attn)
            .at_instances_of(pz.nn.Softmax)
            .apply(lambda softmax: AttentionProfile(wrapped=softmax, horizon=attention_profile_horizon))
        )
    return transformer
