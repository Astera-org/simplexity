import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_nonergodic_hidden_markov_model
from simplexity.generative_processes.nonergodic_state_sampler import NonergodicStateSampler


def test_nonergodic_state_sampler():
    kwargs = {"p": 0.4, "q": 0.25}
    sampler = NonergodicStateSampler(
        process_names=["mr_name", "mr_name"],
        process_kwargs=[kwargs, kwargs],
        process_weights=[0.8, 0.2],
    )
    key = jax.random.PRNGKey(0)
    state = sampler.sample(key)
    assert state.shape == (8,)
    assert jnp.isclose(jnp.sum(state), 1)
    zeros = state == 0
    assert jnp.all(zeros[:4]) or jnp.all(zeros[4:])


def test_integration():
    n_coins = 10
    batch_size = 100
    sequence_len = 100

    process_names = ["coin"] * n_coins
    process_kwargs = [{"p": 0.4}] * n_coins
    process_weights = [1 / n_coins] * n_coins
    vocab_maps = [[2 * i, 2 * i + 1] for i in range(n_coins)]
    state_sampler = NonergodicStateSampler(process_names, process_kwargs, process_weights)
    process = build_nonergodic_hidden_markov_model(process_names, process_kwargs, process_weights, vocab_maps)

    key = jax.random.PRNGKey(0)
    state_key, gen_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, batch_size)
    states = eqx.filter_vmap(state_sampler.sample)(state_keys)
    gen_keys = jax.random.split(gen_key, batch_size)
    _, obs = process.generate(states, gen_keys, sequence_len, False)

    assert obs.shape == (batch_size, sequence_len)

    def one_coin(obs: jax.Array) -> jax.Array:
        min_val = jnp.min(obs)
        max_val = jnp.max(obs)
        return jax.lax.cond(
            min_val == max_val,
            lambda: True,
            lambda: jnp.logical_and(max_val - min_val == 1, jnp.mod(min_val, 2) == 0),
        )

    assert jnp.all(eqx.filter_vmap(one_coin)(obs))
