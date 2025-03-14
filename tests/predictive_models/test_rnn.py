import jax

from simplexity.predictive_models.rnn import RNN


def test_rnn():
    key = jax.random.PRNGKey(0)
    vocab_size = 2
    model = RNN(in_size=vocab_size, out_size=vocab_size, hidden_sizes=[4, 4], key=key)

    sequence_len = 2
    xs = jax.random.randint(key, (sequence_len,), 0, vocab_size)
    one_hot_xs = jax.nn.one_hot(xs, vocab_size)
    ys = model(one_hot_xs)
    assert ys.shape == (sequence_len, vocab_size)
