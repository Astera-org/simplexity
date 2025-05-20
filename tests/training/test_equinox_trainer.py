import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
import optax

from simplexity.predictive_models.gru_rnn import GRURNN
from simplexity.training.equinox_trainer import EquinoxTrainer, loss_fn


def test_equinox_trainer():
    vocab_size = 3
    trainer = EquinoxTrainer.build(
        root_rng=jax.random.PRNGKey(0),
        model=GRURNN(
            in_size=vocab_size,
            out_size=vocab_size,
            hidden_sizes=(10,),
            key=jax.random.PRNGKey(0),
        ),
        optimizer_def=optax.adam(0.1),
        loss_fn=loss_fn,
    )

    initial_params = jax.tree_util.tree_map(lambda x: x.copy(), eqx.filter(trainer.model.value, eqx.is_array))

    batch_size = 8
    sequence_len = 10
    data = jax.random.randint(jax.random.PRNGKey(0), (batch_size, sequence_len), 0, vocab_size)
    inputs = jax.nn.one_hot(data[:, :-1], vocab_size)
    labels = data[:, 1:]

    for step in range(10):
        metrics = trainer.step(inputs=inputs, labels=labels)
        assert metrics["loss"] > 0.0, f"Loss should be positive at step {step}"

    updated_params = eqx.filter(trainer.model.value, eqx.is_array)

    def check_param_changes(old, new):
        diff = jnp.abs(old - new)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)
        assert max_diff > 1e-6, f"Parameters should change by more than 1e-6, got max diff {max_diff}"
        assert mean_diff > 1e-7, f"Parameters should change by more than 1e-7 on average, got mean diff {mean_diff}"
        return True

    result = jax.tree_util.tree_map(check_param_changes, initial_params, updated_params)
    assert result, "Parameters should change"
