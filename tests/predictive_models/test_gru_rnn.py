import equinox as eqx
import jax

from simplexity.predictive_models.gru_rnn import GRURNN


def test_gru_rnn():
    key = jax.random.PRNGKey(0)
    vocab_size = 8
    embedding_size = 16
    model = GRURNN(vocab_size=vocab_size, embedding_size=embedding_size, hidden_sizes=[6, 4], key=key)

    batch_size = 2
    sequence_len = 16
    xs = jax.random.randint(key, (batch_size, sequence_len), 0, vocab_size)
    ys = eqx.filter_vmap(model)(xs)
    assert ys.shape == (batch_size, sequence_len, vocab_size)
