from typing import TypeVar, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.utils import normalize_simplex, stationary_distribution

State = TypeVar("State", bound=jax.Array)


class HiddenMarkovModel(GenerativeProcess[State]):
    """A Hidden Markov Model.

    Parameters
    ----------
    transition_matrices: jax.Array

    T[i,j,k] = P(obs=i, next=j | current=k)
    T[i,j,:] * s = P(obs=i, next=j)
    sum(T[i,:,:] * s) = P(obs=i)
    T[i,:,:] = T[k -> j | i]
    sum(T[:,j,k]) = P(next=j | current=k)
    ? = P(next=j | obs=i, current=k)
    """

    transition_matrices: jax.Array
    log_transition_matrices: jax.Array
    stationary_distribution: jax.Array
    log_stationary_distribution: jax.Array

    def __init__(self, transition_matrices: jax.Array, log: bool = False):
        self.validate_transition_matrices(transition_matrices)
        if log:
            self.transition_matrices = jnp.exp(transition_matrices)
            self.log_transition_matrices = transition_matrices
        else:
            self.transition_matrices = transition_matrices
            self.log_transition_matrices = jnp.log(transition_matrices)

        state_transition_matrix = jnp.sum(self.transition_matrices, axis=0)

        self.stationary_distribution = stationary_distribution(state_transition_matrix)
        self.log_stationary_distribution = jnp.log(self.stationary_distribution)

    def validate_transition_matrices(self, transition_matrices: jax.Array):
        """Validate the transition matrices."""
        if transition_matrices.ndim != 3 or transition_matrices.shape[1] != transition_matrices.shape[2]:
            raise ValueError("Transition matrices must have shape (num_observations, num_states, num_states)")

    @property
    def num_observations(self) -> int:
        """The number of distinct observations that can be emitted by the model."""
        return self.transition_matrices.shape[0]

    @property
    def num_states(self) -> int:
        """The number of hidden states in the model."""
        return self.transition_matrices.shape[1]

    @eqx.filter_jit
    def emit_observation(self, state: State, key: chex.PRNGKey) -> jax.Array:
        """Emit an observation based on the state of the generative process."""
        obs_state_probs = self.transition_matrices @ state
        obs_probs = jnp.sum(obs_state_probs, axis=1)
        obs_probs = normalize_simplex(obs_probs)
        obs = jax.random.choice(key, self.num_observations, p=obs_probs)
        return obs

    @eqx.filter_jit
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Evolve the state of the generative process based on the observation.

        The input state represents a prior distribution over hidden states, and
        the returned state represents a posterior distribution over hidden states
        conditioned on the observation.
        """
        next_state = self.transition_matrices[obs] @ state
        next_state = normalize_simplex(next_state)
        return cast(State, next_state)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""

        def _scan_fn(state_distribution, observation):
            return self.transition_matrices[observation] @ state_distribution, None

        state_distribution, _ = jax.lax.scan(_scan_fn, init=self.stationary_distribution, xs=observations)

        return jnp.sum(state_distribution)

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""

        def _scan_fn(log_state_distribution, observation):
            return jax.nn.logsumexp(self.log_transition_matrices[observation] + log_state_distribution, axis=1), None

        log_state_distribution, _ = jax.lax.scan(_scan_fn, init=self.log_stationary_distribution, xs=observations)

        return jax.nn.logsumexp(log_state_distribution)
