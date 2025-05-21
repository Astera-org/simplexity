import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
import optax

from simplexity.predictive_models.gru_rnn import GRURNN
from simplexity.training.equinox_trainer import EquinoxTrainer, loss_fn


def generate_data_batch(batch_size: int, sequence_len: int, vocab_size: int, seed: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    data = jax.random.randint(key, (batch_size, sequence_len), 0, vocab_size)
    inputs = jax.nn.one_hot(data[:, :-1], vocab_size)
    labels = data[:, 1:]
    return inputs, labels


def test_equinox_trainer():
    vocab_size = 3
    batch_size = 8
    sequence_len = 10

    model = GRURNN(
        in_size=vocab_size,
        out_size=vocab_size,
        hidden_sizes=(10,),
        key=jax.random.PRNGKey(0),
    )

    trainer = EquinoxTrainer.build(
        root_rng=jax.random.PRNGKey(0),
        model=model,
        optimizer_def=optax.adam(0.1),
        loss_fn=loss_fn,
    )

    inputs_0, _ = generate_data_batch(batch_size, sequence_len, vocab_size, seed=0)

    initial_params = jax.tree_util.tree_map(lambda x: x.copy(), eqx.filter(trainer.model.value, eqx.is_array))
    initial_outputs = eqx.filter_vmap(model)(inputs_0)

    for step in range(1, 4):
        inputs, labels = generate_data_batch(batch_size, sequence_len, vocab_size, seed=step)
        metrics = trainer.step(inputs=inputs, labels=labels)
        assert metrics["loss"] > 0.0, f"Loss should be positive at step {step}"

    updated_params = eqx.filter(trainer.model.value, eqx.is_array)
    updated_outputs = eqx.filter_vmap(trainer.model.value)(inputs_0)

    def assert_changed(old, new):
        diff = jnp.abs(old - new)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)
        assert max_diff > 1e-6, f"Parameters should change by more than 1e-6, got max diff {max_diff}"
        assert mean_diff > 1e-7, f"Parameters should change by more than 1e-7 on average, got mean diff {mean_diff}"
        return True

    assert jax.tree_util.tree_map(assert_changed, initial_params, updated_params)
    assert jax.tree_util.tree_map(assert_changed, initial_outputs, updated_outputs)
