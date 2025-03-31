import jax
import pytest
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.penzai_utils import (
    NamedParameters,
    ParameterTree,
    calculate_llamalike_transformer_parameter_count,
    get_parameter_count,
    get_parameter_list,
    get_parameter_tree,
)

CONFIG = LlamalikeTransformerConfig(
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

PARAMS = {
    "embeddings": CONFIG.vocab_size * CONFIG.embedding_dim,
    "layer_norm": CONFIG.embedding_dim,
    "queries": CONFIG.embedding_dim * CONFIG.num_kv_heads * CONFIG.query_head_multiplier * CONFIG.projection_dim,
    "keys": CONFIG.embedding_dim * CONFIG.num_kv_heads * CONFIG.projection_dim,
    "values": CONFIG.embedding_dim * CONFIG.num_kv_heads * CONFIG.projection_dim,
    "outputs": CONFIG.embedding_dim * CONFIG.num_kv_heads * CONFIG.query_head_multiplier * CONFIG.projection_dim,
    "mlp_gate": CONFIG.embedding_dim * CONFIG.mlp_hidden_dim,
    "mlp_value": CONFIG.embedding_dim * CONFIG.mlp_hidden_dim,
    "mlp_output": CONFIG.mlp_hidden_dim * CONFIG.embedding_dim,
    "lm_head": CONFIG.embedding_dim * CONFIG.vocab_size,
}
PARAMS["attention"] = PARAMS["queries"] + PARAMS["keys"] + PARAMS["values"] + PARAMS["outputs"]
PARAMS["mlp"] = PARAMS["mlp_gate"] + PARAMS["mlp_value"] + PARAMS["mlp_output"]
PARAMS["decoder_block"] = PARAMS["layer_norm"] + PARAMS["attention"] + PARAMS["layer_norm"] + PARAMS["mlp"]
PARAMS["total"] = (
    PARAMS["embeddings"]
    + PARAMS["decoder_block"] * CONFIG.num_decoder_blocks
    + PARAMS["layer_norm"]
    + PARAMS["lm_head"]
)


@pytest.fixture
def transformer():
    base_rng = jax.random.PRNGKey(0)
    return build_llamalike_transformer(CONFIG, init_base_rng=base_rng)


def test_get_parameter_count_from_config():
    assert calculate_llamalike_transformer_parameter_count(CONFIG) == PARAMS["total"]


def test_get_parameter_count(transformer):
    assert get_parameter_count(transformer) == PARAMS["total"]


def test_get_parameter_tree(transformer):
    actual = get_parameter_tree(transformer)

    layer_norm = ParameterTree(
        name="RMSLayerNorm.Linear",
        parameters=PARAMS["layer_norm"],
        children=[],
    )

    decoder_block = ParameterTree(
        name="TransformerBlock",
        parameters=PARAMS["decoder_block"],
        children=[
            ParameterTree(
                name="Residual.Sequential",
                parameters=PARAMS["layer_norm"] + PARAMS["attention"],
                children=[
                    layer_norm,
                    ParameterTree(
                        name="Attention",
                        parameters=PARAMS["attention"],
                        children=[
                            ParameterTree(
                                name="Sequential.Linear",
                                parameters=PARAMS["queries"],
                                children=[],
                            ),
                            ParameterTree(
                                name="Sequential.Linear",
                                parameters=PARAMS["keys"],
                                children=[],
                            ),
                            ParameterTree(
                                name="Sequential.Linear",
                                parameters=PARAMS["values"],
                                children=[],
                            ),
                            ParameterTree(
                                name="Sequential.Linear",
                                parameters=PARAMS["outputs"],
                                children=[],
                            ),
                        ],
                    ),
                ],
            ),
            ParameterTree(
                name="Residual.Sequential",
                parameters=PARAMS["layer_norm"] + PARAMS["mlp"],
                children=[
                    layer_norm,
                    ParameterTree(
                        name="TransformerFeedForward",
                        parameters=PARAMS["mlp"],
                        children=[
                            ParameterTree(
                                name="BranchAndMultiplyTogether",
                                parameters=PARAMS["mlp_gate"] + PARAMS["mlp_value"],
                                children=[
                                    ParameterTree(
                                        name="NamedGroup.Linear",
                                        parameters=PARAMS["mlp_gate"],
                                        children=[],
                                    ),
                                    ParameterTree(
                                        name="Linear",
                                        parameters=PARAMS["mlp_value"],
                                        children=[],
                                    ),
                                ],
                            ),
                            ParameterTree(
                                name="Linear",
                                parameters=PARAMS["mlp_output"],
                                children=[],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    expected = ParameterTree(
        name="TransformerLM.Sequential",
        parameters=PARAMS["total"],
        children=[
            ParameterTree(
                name="EmbeddingLookup.EmbeddingTable",
                parameters=PARAMS["embeddings"],
                children=[],
            )
        ]
        + [decoder_block] * CONFIG.num_decoder_blocks
        + [layer_norm]
        + [
            ParameterTree(
                name="Linear",
                parameters=PARAMS["lm_head"],
                children=[],
            )
        ],
    )
    assert actual == expected


def test_get_parameter_list(transformer):
    tree = get_parameter_tree(transformer)
    actual = get_parameter_list(tree)
    decoder_block = [
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.RMSLayerNorm.Linear",
            parameters=PARAMS["layer_norm"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.Attention.Sequential.Linear",
            parameters=PARAMS["queries"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.Attention.Sequential.Linear",
            parameters=PARAMS["keys"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.Attention.Sequential.Linear",
            parameters=PARAMS["values"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.Attention.Sequential.Linear",
            parameters=PARAMS["outputs"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.RMSLayerNorm.Linear",
            parameters=PARAMS["layer_norm"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.TransformerFeedForward.BranchAndMultiplyTogether.NamedGroup.Linear",
            parameters=PARAMS["mlp_gate"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.TransformerFeedForward.BranchAndMultiplyTogether.Linear",
            parameters=PARAMS["mlp_value"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.TransformerBlock.Residual.Sequential.TransformerFeedForward.Linear",
            parameters=PARAMS["mlp_output"],
        ),
    ]
    expected = [
        NamedParameters(
            name="TransformerLM.Sequential.EmbeddingLookup.EmbeddingTable",
            parameters=PARAMS["embeddings"],
        ),
    ]
    expected += decoder_block * CONFIG.num_decoder_blocks
    expected += [
        NamedParameters(
            name="TransformerLM.Sequential.RMSLayerNorm.Linear",
            parameters=PARAMS["layer_norm"],
        ),
        NamedParameters(
            name="TransformerLM.Sequential.Linear",
            parameters=PARAMS["lm_head"],
        ),
    ]
    assert actual == expected
