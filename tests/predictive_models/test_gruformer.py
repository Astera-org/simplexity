import jax
from penzai import pz
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig

from simplexity.predictive_models.gruformer import build_llamalike_gruformer


def test_gruformer():
    vocab_size = 8
    config = LlamalikeTransformerConfig(
        num_kv_heads=2,
        query_head_multiplier=2,
        embedding_dim=16,
        projection_dim=16,
        mlp_hidden_dim=16,
        num_decoder_blocks=2,
        vocab_size=vocab_size,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=True,
    )
    model = build_llamalike_gruformer(config, init_base_rng=jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(0)
    batch_size = 13
    sequence_length = 31
    sequences = jax.random.randint(key, (batch_size, sequence_length), 0, vocab_size)
    inputs = pz.nx.wrap(sequences, "batch", "seq")
    outputs = model(inputs)
    assert isinstance(outputs, pz.nx.NamedArray)  # type: ignore
    assert outputs.named_axes == {
        "batch": batch_size,
        "seq": sequence_length,
        "vocabulary": vocab_size,
    }
