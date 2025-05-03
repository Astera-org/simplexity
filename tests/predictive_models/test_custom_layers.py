import equinox as eqx
import jax
from penzai import pz

from simplexity.predictive_models.custom_layers import SaveInputs
from simplexity.predictive_models.gru_rnn import GRURNN


def test_save_inputs():
    key = jax.random.PRNGKey(0)
    in_size = 8
    out_size = 2
    hidden_sizes = [6, 4]
    model = GRURNN(in_size=in_size, out_size=out_size, hidden_sizes=hidden_sizes, key=key)

    activations = pz.StateVariable(value=[], label="activations")
    saving_model = pz.select(model).at_instances_of(eqx.nn.Lambda).insert_after(SaveInputs(activations))

    sequence_len = 16
    xs = jax.random.randint(key, (sequence_len, in_size), 0, 2)
    saving_model(xs)
    assert len(activations.value) == len(hidden_sizes) + 1
    assert activations.value[0].shape == (sequence_len, hidden_sizes[0])
    assert activations.value[1].shape == (sequence_len, hidden_sizes[1])
    assert activations.value[-1].shape == (sequence_len, out_size)
