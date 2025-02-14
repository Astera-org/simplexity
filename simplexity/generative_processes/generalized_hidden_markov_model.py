from typing import TypeVar, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.utils import stationary_distribution

State = TypeVar("State", bound=jax.Array)


class GeneralizedHiddenMarkovModel(GenerativeProcess[State]):
    """A Generalized Hidden Markov Model."""

    transition_matrices: jax.Array
    log_transition_matrices: jax.Array
    right_eigenvector: jax.Array
    log_right_eigenvector: jax.Array
    left_eigenvector: jax.Array
    log_left_eigenvector: jax.Array
    _normalizing_constant: jax.Array
    _log_normalizing_constant: jax.Array

    def __init__(self, transition_matrices: jax.Array, log: bool = False):
        self.validate_transition_matrices(transition_matrices)
        if log:
            self.transition_matrices = jnp.exp(transition_matrices)
            self.log_transition_matrices = transition_matrices
        else:
            self.transition_matrices = transition_matrices
            self.log_transition_matrices = jnp.log(transition_matrices)

        state_transition_matrix = jnp.sum(self.transition_matrices, axis=0)

        self.right_eigenvector = stationary_distribution(state_transition_matrix)
        self.log_right_eigenvector = jnp.log(self.right_eigenvector)

        self.left_eigenvector = stationary_distribution(state_transition_matrix.T)
        self.log_left_eigenvector = jnp.log(self.left_eigenvector)

        self._normalizing_constant = self.left_eigenvector @ self.right_eigenvector
        self._log_normalizing_constant = jax.nn.logsumexp(self.log_left_eigenvector + self.log_right_eigenvector)

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
        obs_probs = (self.left_eigenvector @ self.transition_matrices @ state) / (self.left_eigenvector @ state)
        return jax.random.choice(key, self.num_observations, p=obs_probs)

    @eqx.filter_jit
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Evolve the state of the generative process based on the observation.

        The input state represents a prior distribution over hidden states, and
        the returned state represents a posterior distribution over hidden states
        conditioned on the observation.
        """
        return cast(State, self.transition_matrices[obs] @ state)

    @eqx.filter_jit
    def state_probability(self, state: State) -> jax.Array:
        """Compute the probability distribution over states from a state vector."""
        return self.left_eigenvector * state / (self.left_eigenvector @ state)

    @eqx.filter_jit
    def state_log_probability(self, log_state: jax.Array) -> jax.Array:
        """Compute the log probability distribution over states from a log state vector."""
        return self.log_left_eigenvector + log_state - jax.nn.logsumexp(self.log_left_eigenvector + log_state)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""

        def _scan_fn(right_vector, observation):
            return self.transition_matrices[observation] @ right_vector, None

        right_vector, _ = jax.lax.scan(_scan_fn, init=self.right_eigenvector, xs=observations)
        return (self.left_eigenvector @ right_vector) / self._normalizing_constant

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""

        def _scan_fn(log_right_vector, observation):
            return jax.nn.logsumexp(self.log_transition_matrices[observation] + log_right_vector, axis=1), None

        log_right_vector, _ = jax.lax.scan(_scan_fn, init=self.log_right_eigenvector, xs=observations)
        return jax.nn.logsumexp(self.log_left_eigenvector + log_right_vector) - self._log_normalizing_constant
