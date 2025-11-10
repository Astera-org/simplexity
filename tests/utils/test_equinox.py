import equinox as eqx
import jax

from simplexity.predictive_models.gru_rnn import build_gru_rnn
from simplexity.utils.equinox import vmap_model


def test_vmap_model():
    vocab_size = 2
    model = build_gru_rnn(vocab_size=vocab_size, embedding_size=2, num_layers=1, hidden_size=2, seed=0)

    def f(model: eqx.Module, inputs: jax.Array) -> jax.Array:
        return model(inputs)

    vmap_f = vmap_model(f)
    batch_size = 2
    sequence_len = 2
    inputs = jax.random.randint(jax.random.key(0), (batch_size, sequence_len), 0, vocab_size)
    outputs = vmap_f(model=model, inputs=inputs)
    assert outputs.shape == (batch_size, sequence_len, vocab_size)
