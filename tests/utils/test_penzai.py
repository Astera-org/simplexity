import dataclasses

import jax
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.utils.penzai import ParamCountNode, get_parameter_count_tree


def test_get_parameter_count_tree():
    config = LlamalikeTransformerConfig(
        embedding_dim=2,
        num_decoder_blocks=3,
        num_kv_heads=5,
        projection_dim=7,
        query_head_multiplier=11,
        mlp_hidden_dim=13,
        vocab_size=17,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    base_rng = jax.random.PRNGKey(0)
    transformer = build_llamalike_transformer(config, init_base_rng=base_rng)
    actual = get_parameter_count_tree(transformer)

    param_counts = {
        "embeddings": config.vocab_size * config.embedding_dim,
        "layer_norm": config.embedding_dim,
        "queries": config.embedding_dim * config.num_kv_heads * config.query_head_multiplier * config.projection_dim,
        "keys": config.embedding_dim * config.num_kv_heads * config.projection_dim,
        "values": config.embedding_dim * config.num_kv_heads * config.projection_dim,
        "outputs": config.embedding_dim * config.num_kv_heads * config.query_head_multiplier * config.projection_dim,
        "mlp_gate": config.embedding_dim * config.mlp_hidden_dim,
        "mlp_value": config.embedding_dim * config.mlp_hidden_dim,
        "mlp_output": config.mlp_hidden_dim * config.embedding_dim,
        "lm_head": config.embedding_dim * config.vocab_size,
    }
    param_counts["attention"] = (
        param_counts["queries"] + param_counts["keys"] + param_counts["values"] + param_counts["outputs"]
    )
    param_counts["mlp"] = param_counts["mlp_gate"] + param_counts["mlp_value"] + param_counts["mlp_output"]
    param_counts["decoder_block"] = (
        param_counts["layer_norm"] + param_counts["attention"] + param_counts["layer_norm"] + param_counts["mlp"]
    )
    param_counts["transformer"] = (
        param_counts["embeddings"]
        + param_counts["decoder_block"] * config.num_decoder_blocks
        + param_counts["layer_norm"]
        + param_counts["lm_head"]
    )

    decoder_block = ParamCountNode(
        name="block_{block_index}",
        param_count=param_counts["decoder_block"],
        children=[
            ParamCountNode(
                name="pre_attention_norm",
                param_count=param_counts["layer_norm"],
                children=[
                    ParamCountNode(
                        name="scale.weights",
                        param_count=param_counts["layer_norm"],
                        children=[],
                    )
                ],
            ),
            ParamCountNode(
                name="attention",
                param_count=param_counts["attention"],
                children=[
                    ParamCountNode(
                        name="query.weights",
                        param_count=param_counts["queries"],
                        children=[],
                    ),
                    ParamCountNode(
                        name="key.weights",
                        param_count=param_counts["keys"],
                        children=[],
                    ),
                    ParamCountNode(
                        name="value.weights",
                        param_count=param_counts["values"],
                        children=[],
                    ),
                    ParamCountNode(
                        name="output.weights",
                        param_count=param_counts["outputs"],
                        children=[],
                    ),
                ],
            ),
            ParamCountNode(
                name="pre_ffw_norm",
                param_count=param_counts["layer_norm"],
                children=[
                    ParamCountNode(
                        name="scale.weights",
                        param_count=param_counts["layer_norm"],
                        children=[],
                    )
                ],
            ),
            ParamCountNode(
                name="mlp",
                param_count=param_counts["mlp"],
                children=[
                    ParamCountNode(
                        name="gating_linear.weights",
                        param_count=param_counts["mlp_gate"],
                        children=[],
                    ),
                    ParamCountNode(
                        name="value_linear.weights",
                        param_count=param_counts["mlp_value"],
                        children=[],
                    ),
                    ParamCountNode(
                        name="out_linear.weights",
                        param_count=param_counts["mlp_output"],
                        children=[],
                    ),
                ],
            ),
        ],
    )

    expected = ParamCountNode(
        name="transformer",
        param_count=param_counts["transformer"],
        children=[
            ParamCountNode(
                name="embedder.embeddings",
                param_count=param_counts["embeddings"],
                children=[],
            ),
        ]
        + [dataclasses.replace(decoder_block, name=f"block_{i}") for i in range(config.num_decoder_blocks)]
        + [
            ParamCountNode(
                name="final_norm",
                param_count=param_counts["layer_norm"],
                children=[
                    ParamCountNode(
                        name="scale.weights",
                        param_count=param_counts["layer_norm"],
                        children=[],
                    ),
                ],
            ),
            ParamCountNode(
                name="lm_head.weights",
                param_count=param_counts["lm_head"],
                children=[],
            ),
        ],
    )
    assert actual == expected
