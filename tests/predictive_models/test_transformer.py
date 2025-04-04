import jax
from penzai import pz
from simplexity.predictive_models.transformer import build_transformer


def test_transformer():
    vocab_size = 10
    transformer = build_transformer(
        num_kv_heads=2,
        query_head_multiplier=2,
        embedding_dim=16,
        projection_dim=16,
        mlp_hidden_dim=16,
        num_decoder_blocks=2,
        vocab_size=10,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=True,
        attention_profile_horizon=50,
        seed=42,
    )

    batch_size = 10
    sequence_len = 100
    key = jax.random.key(0)
    xs = jax.random.randint(key, (batch_size, sequence_len), 0, vocab_size)
    xs = pz.nx.wrap(xs, "batch", "seq")
    ys = transformer(xs)
    expect_axes = ("batch", "seq", "vocabulary")
    actual_axes = tuple(ys.named_axes.keys())
    assert expect_axes == actual_axes, f"{expect_axes=} != {actual_axes=}"
    assert ys.named_axes["batch"] == batch_size
    assert ys.named_axes["seq"] == sequence_len
    assert ys.named_axes["vocabulary"] == vocab_size
