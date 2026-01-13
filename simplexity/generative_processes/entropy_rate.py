"""Entropy rate computation for generative processes."""

import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess


def compute_entropy_rate(
    process: GenerativeProcess,
    num_steps: int = 50000,
    seed: int = 42,
) -> float:
    """Compute the entropy rate of a generative process by simulation.

    The entropy rate is the optimal cross-entropy loss achievable by any model
    trained on sequences from this process.

    Args:
        process: The generative process to compute entropy rate for.
        num_steps: Number of simulation steps for estimation.
        seed: Random seed for reproducibility.

    Returns:
        Entropy rate in nats (natural log units).
    """
    key = jax.random.PRNGKey(seed)
    state = process.initial_state
    total_entropy = 0.0

    for _ in range(num_steps):
        key, subkey = jax.random.split(key)
        obs_dist = process.observation_probability_distribution(state)

        eps = 1e-10
        p_safe = jnp.clip(obs_dist, eps, 1.0)
        entropy = -jnp.sum(jnp.where(obs_dist > eps, obs_dist * jnp.log(p_safe), 0.0))
        total_entropy += float(entropy)

        obs = jax.random.categorical(subkey, jnp.log(p_safe))
        state = process.transition_states(state, obs)

    return total_entropy / num_steps


def compute_entropy_rate_bits(
    process: GenerativeProcess,
    num_steps: int = 50000,
    seed: int = 42,
) -> float:
    """Compute the entropy rate in bits (log base 2).

    Args:
        process: The generative process to compute entropy rate for.
        num_steps: Number of simulation steps for estimation.
        seed: Random seed for reproducibility.

    Returns:
        Entropy rate in bits.
    """
    return compute_entropy_rate(process, num_steps, seed) / float(jnp.log(2))
