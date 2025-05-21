import dataclasses

import jax
import pytest
from penzai import pz
from penzai.core.variables import UnboundVariableError
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.predictive_models.predictive_model import PredictiveModel
from simplexity.utils.penzai import (
    ParamCountNode,
    PenzaiWrapper,
    VariableLabelClass,
    VariableValueClass,
    deconstruct_variables,
    get_parameter_count_tree,
    reconstruct_variables,
    use_penzai_model,
)


def test_penzai_wrapper():
    vocab_size = 4
    batch_size = 2
    seq_length = 16

    config = LlamalikeTransformerConfig(
        embedding_dim=8,
        num_decoder_blocks=1,
        num_kv_heads=1,
        projection_dim=8,
        query_head_multiplier=1,
        mlp_hidden_dim=8,
        vocab_size=vocab_size,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    base_rng = jax.random.PRNGKey(0)
    transformer = build_llamalike_transformer(config, init_base_rng=base_rng)

    wrapped_model = PenzaiWrapper(transformer)
    inputs = jax.random.randint(jax.random.key(0), (batch_size, seq_length), 0, vocab_size)
    assert isinstance(inputs, jax.Array)
    outputs = wrapped_model(inputs)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (batch_size, seq_length, vocab_size)


def test_use_penzai_model():
    vocab_size = 4
    batch_size = 2
    seq_length = 16

    config = LlamalikeTransformerConfig(
        embedding_dim=8,
        num_decoder_blocks=1,
        num_kv_heads=1,
        projection_dim=8,
        query_head_multiplier=1,
        mlp_hidden_dim=8,
        vocab_size=vocab_size,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    base_rng = jax.random.PRNGKey(0)
    transformer = build_llamalike_transformer(config, init_base_rng=base_rng)

    inputs = jax.random.randint(jax.random.key(0), (batch_size, seq_length), 0, vocab_size)
    assert isinstance(inputs, jax.Array)

    @use_penzai_model
    def f(model: PredictiveModel, inputs: jax.Array) -> jax.Array:
        return model(inputs)

    outputs = f(model=transformer, inputs=inputs)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (batch_size, seq_length, vocab_size)


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


CONFIG = LlamalikeTransformerConfig(
    num_kv_heads=1,
    query_head_multiplier=1,
    embedding_dim=32,
    projection_dim=32,
    mlp_hidden_dim=32,
    num_decoder_blocks=1,
    vocab_size=32,
    mlp_variant="geglu_approx",
    tie_embedder_and_logits=False,
)
DECONSTRUCTED_VARIABLES = {
    "axis_names": (
        ("vocabulary", "embedding"),
        ("embedding",),
        ("embedding", "heads", "projection"),
        ("embedding", "heads", "projection"),
        ("embedding", "heads", "projection"),
        ("heads", "projection", "embedding"),
        ("embedding",),
        ("embedding", "neurons"),
        ("embedding", "neurons"),
        ("neurons", "embedding"),
        ("embedding",),
        ("embedding", "vocabulary"),
    ),
    "axis_sizes": (
        (32, 32),
        (32,),
        (32, 1, 32),
        (32, 1, 32),
        (32, 1, 32),
        (1, 32, 32),
        (32,),
        (32, 32),
        (32, 32),
        (32, 32),
        (32,),
        (32, 32),
    ),
    "variable_value_classes": (
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
        VariableValueClass.PARAMETER,
    ),
    "variable_labels": (
        "transformer/embedder.embeddings",
        "transformer/block_0/pre_attention_norm/scale.weights",
        "transformer/block_0/attention/query.weights",
        "transformer/block_0/attention/key.weights",
        "transformer/block_0/attention/value.weights",
        "transformer/block_0/attention/output.weights",
        "transformer/block_0/pre_ffw_norm/scale.weights",
        "transformer/block_0/mlp/gating_linear.weights",
        "transformer/block_0/mlp/value_linear.weights",
        "transformer/block_0/mlp/out_linear.weights",
        "transformer/final_norm/scale.weights",
        "transformer/lm_head.weights",
    ),
    "variable_label_classes": (
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
        VariableLabelClass.STR,
    ),
    "metadata": tuple({} for _ in range(12)),
}


def test_deconstruct_variables():
    base_rng = jax.random.PRNGKey(0)
    transformer = build_llamalike_transformer(CONFIG, init_base_rng=base_rng)
    _, variables = pz.unbind_variables(transformer, freeze=True)
    actual = deconstruct_variables(variables)
    for key in DECONSTRUCTED_VARIABLES:
        assert actual[key] == DECONSTRUCTED_VARIABLES[key]


def test_reconstruct_variables():
    sequences = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, CONFIG.vocab_size)
    inputs = pz.nx.wrap(sequences, "batch", "seq")
    unbound_transformer = build_llamalike_transformer(CONFIG)
    with pytest.raises(UnboundVariableError):
        unbound_transformer(inputs)
    DECONSTRUCTED_VARIABLES["data_arrays"] = tuple(
        jax.random.normal(jax.random.PRNGKey(i), shape) for i, shape in enumerate(DECONSTRUCTED_VARIABLES["axis_sizes"])
    )
    variables = reconstruct_variables(DECONSTRUCTED_VARIABLES)
    transformer = pz.bind_variables(unbound_transformer, variables)
    transformer(inputs)
