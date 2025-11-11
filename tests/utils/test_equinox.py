import equinox as eqx
import jax

from simplexity.utils.equinox import vmap_model


def test_vmap_model():
    """Test vmap_model function."""
    vocab_size = 2
    sequence_len = 16
    model = eqx.nn.Linear(in_features=sequence_len, out_features=sequence_len, key=jax.random.key(0))

    def f(model: eqx.nn.Linear, inputs: jax.Array) -> jax.Array:
        return model(inputs)

    vmap_f = vmap_model(f)
    batch_size = 2
    inputs = jax.random.randint(jax.random.key(0), (batch_size, sequence_len), 0, vocab_size)
    outputs = vmap_f(model=model, inputs=inputs)
    assert outputs.shape == (batch_size, sequence_len)
