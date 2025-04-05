import jax
from penzai.models.transformer.variants.llamalike_common import LlamalikeTransformerConfig, build_llamalike_transformer

from simplexity.configs.validation.config import Config as ValidateConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.validation.validate_penzai_model import validate


def test_validate():
    cfg = ValidateConfig(seed=0, sequence_len=4, batch_size=2, num_steps=3, log_every=5)
    data_generator = build_hidden_markov_model("even_ones", p=0.5)
    config = LlamalikeTransformerConfig(
        num_kv_heads=1,
        query_head_multiplier=1,
        embedding_dim=2,
        projection_dim=2,
        mlp_hidden_dim=2,
        num_decoder_blocks=1,
        vocab_size=data_generator.vocab_size,
        mlp_variant="geglu_approx",
        tie_embedder_and_logits=False,
    )
    model = build_llamalike_transformer(config, init_base_rng=jax.random.PRNGKey(0))
    metrics = validate(model, cfg, data_generator)
    assert metrics["loss"] > 0.0
