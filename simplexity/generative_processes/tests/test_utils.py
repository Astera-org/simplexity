import jax
import jax.numpy as jnp

from simplexity.generative_processes.utils import log_matmul


def test_log_matmul():
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    A = jax.random.uniform(key_a, (3, 4))
    B = jax.random.uniform(key_b, (4, 5))
    assert jnp.allclose(log_matmul(jnp.log(A), jnp.log(B)), jnp.log(A @ B))
