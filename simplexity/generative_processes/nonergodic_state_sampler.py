from collections.abc import Mapping, Sequence
from typing import Any

import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_transition_matrices
from simplexity.generative_processes.state_sampler import StateSampler
from simplexity.generative_processes.transition_matrices import (
    HMM_MATRIX_FUNCTIONS,
    stationary_state,
)


class NonergodicStateSampler(StateSampler):
    """A sampler for the state of a nonergodic process."""

    states: jax.Array
    probabilities: jax.Array

    def __init__(
        self,
        process_names: list[str],
        process_kwargs: Sequence[Mapping[str, Any]],
        process_weights: Sequence[float],
    ):
        stationary_distributions: list[jax.Array] = []
        for process_name, kwargs in zip(process_names, process_kwargs, strict=True):
            transition_matrix = build_transition_matrices(HMM_MATRIX_FUNCTIONS, process_name, **kwargs)
            state_transition_matrix = transition_matrix.sum(axis=0)
            stationary_distribution = stationary_state(state_transition_matrix.T)
            stationary_distributions.append(stationary_distribution)
        num_states = len(process_names)
        state_size = sum(distribution.shape[0] for distribution in stationary_distributions)
        states = jnp.zeros((num_states, state_size))
        offset = 0
        for i, distribution in enumerate(stationary_distributions):
            states = states.at[i, offset : offset + distribution.shape[0]].set(distribution)
            offset += distribution.shape[0]
        self.states = states
        weights = jnp.array(process_weights)
        assert jnp.all(weights >= 0)
        self.probabilities = weights / jnp.sum(weights)

    def sample(self, key: chex.PRNGKey) -> jax.Array:
        """Randomly sample a state from the nonergodic process."""
        return jax.random.choice(key, self.states, p=self.probabilities)
