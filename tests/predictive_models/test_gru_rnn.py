import jax

from simplexity.predictive_models.gru_rnn import GRURNN


def test_gru_rnn():
    key = jax.random.PRNGKey(0)
    in_size = 8
    out_size = 2
    model = GRURNN(in_size=in_size, out_size=out_size, hidden_sizes=[6, 4], key=key)

    sequence_len = 16
    xs = jax.random.randint(key, (sequence_len, in_size), 0, 2)
    ys = model(xs)
    assert ys.shape == (sequence_len, out_size)
