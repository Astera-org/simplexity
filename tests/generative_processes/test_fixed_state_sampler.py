import jax
import jax.numpy as jnp

from simplexity.generative_processes.fixed_state_sampler import FixedStateSampler


def test_fixed_state_sampler():
    state = jnp.array([0, 1, 0, 1])
    sampler = FixedStateSampler(state)
    key = jax.random.PRNGKey(0)
    sampled_state = sampler.sample(key)
    assert jnp.all(sampled_state == state)
