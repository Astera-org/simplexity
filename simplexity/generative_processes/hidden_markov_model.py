from typing import TypeVar, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.utils import normalize_simplex

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
    _log_transition_matrices: jax.Array
    right_stationary_distribution: jax.Array
    _log_right_stationary_distribution: jax.Array
    left_stationary_distribution: jax.Array
    _log_left_stationary_distribution: jax.Array

    def __init__(self, transition_matrices: jax.Array, log: bool = False):
        if log:
            self.transition_matrices = jnp.exp(transition_matrices)
            self._log_transition_matrices = transition_matrices
        else:
            self.transition_matrices = transition_matrices
            self._log_transition_matrices = jnp.log(transition_matrices)

        state_transition_matrix = jnp.sum(self.transition_matrices, axis=0)

        eigenvalues, eigenvectors = jnp.linalg.eig(state_transition_matrix)
        right_stationary_distribution = eigenvectors[:, jnp.isclose(eigenvalues, 1)].real
        self.right_stationary_distribution = normalize_simplex(right_stationary_distribution)
        self._log_right_stationary_distribution = jnp.log(self.right_stationary_distribution)

        eigenvalues, eigenvectors = jnp.linalg.eig(state_transition_matrix.T)
        left_stationary_distribution = eigenvectors[:, jnp.isclose(eigenvalues, 1)].real.T
        self.left_stationary_distribution = normalize_simplex(left_stationary_distribution)
        self._log_left_stationary_distribution = jnp.log(self.left_stationary_distribution)

    def __post_init__(self):
        if self.transition_matrices.ndim != 3 or self.transition_matrices.shape[1] != self.transition_matrices.shape[2]:
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
    def transition(self, state: State, key: chex.PRNGKey) -> tuple[State, jax.Array]:
        """Transition the state of the GHMM, state is a probability distribution over states."""
        obs_state_probs = self.transition_matrices @ state
        obs_probs = jnp.sum(obs_state_probs, axis=1)
        obs_probs = normalize_simplex(obs_probs)
        obs = jax.random.choice(key, self.num_observations, p=obs_probs)
        next_state = self.transition_matrices[obs] @ state
        next_state = normalize_simplex(next_state)
        return cast(State, next_state), obs

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""

        def _scan_fn(state_distribution, observation):
            return self.transition_matrices[observation] @ state_distribution, None

        state_distribution, _ = jax.lax.scan(_scan_fn, init=self.right_stationary_distribution, xs=observations)

        return jnp.sum(state_distribution)

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""

        def _scan_fn(log_state_distribution, observation):
            return jax.nn.logsumexp(self._log_transition_matrices[observation] + log_state_distribution, axis=1), None

        log_state_distribution, _ = jax.lax.scan(
            _scan_fn, init=self._log_right_stationary_distribution, xs=observations
        )

        return jax.nn.logsumexp(log_state_distribution)
