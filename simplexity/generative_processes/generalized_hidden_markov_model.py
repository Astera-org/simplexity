from typing import TypeVar, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess

State = TypeVar("State", bound=jax.Array)


class GeneralizedHiddenMarkovModel(GenerativeProcess[State]):
    """A Generalized Hidden Markov Model."""

    transition_matrices: jax.Array
    log_transition_matrices: jax.Array
    normalizing_eigenvector: jax.Array
    log_normalizing_eigenvector: jax.Array
    state_eigenvector: jax.Array
    log_state_eigenvector: jax.Array
    _normalizing_constant: jax.Array
    _log_normalizing_constant: jax.Array

    def __init__(self, transition_matrices: jax.Array):
        self.validate_transition_matrices(transition_matrices)

        state_transition_matrix = jnp.sum(transition_matrices, axis=0)
        eigenvalues, right_eigenvectors = jnp.linalg.eig(state_transition_matrix)
        principal_eigenvalue = jnp.max(eigenvalues)

        if jnp.isclose(principal_eigenvalue, 1):
            self.transition_matrices = transition_matrices
        else:
            self.transition_matrices = transition_matrices / principal_eigenvalue
        self.log_transition_matrices = jnp.log(transition_matrices)

        normalizing_eigenvector = right_eigenvectors[:, jnp.isclose(eigenvalues, principal_eigenvalue)].squeeze().real
        self.normalizing_eigenvector = normalizing_eigenvector / jnp.sum(normalizing_eigenvector) * self.num_states
        self.log_normalizing_eigenvector = jnp.log(self.normalizing_eigenvector)

        eigenvalues, left_eigenvectors = jnp.linalg.eig(state_transition_matrix.T)
        state_eigenvector = left_eigenvectors[:, jnp.isclose(eigenvalues, principal_eigenvalue)].squeeze().real
        self.state_eigenvector = state_eigenvector / jnp.sum(state_eigenvector)
        self.log_state_eigenvector = jnp.log(self.state_eigenvector)

        self._normalizing_constant = self.state_eigenvector @ self.normalizing_eigenvector
        self._log_normalizing_constant = jax.nn.logsumexp(self.log_state_eigenvector + self.log_normalizing_eigenvector)

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
        obs_probs = self.observation_probability_distribution(state)
        return jax.random.choice(key, self.num_observations, p=obs_probs)

    @eqx.filter_jit
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Evolve the state of the generative process based on the observation.

        The input state represents a prior distribution over hidden states, and
        the returned state represents a posterior distribution over hidden states
        conditioned on the observation.
        """
        state = cast(State, state @ self.transition_matrices[obs])
        return cast(State, state / (state @ self.normalizing_eigenvector))

    @eqx.filter_jit
    def normalize_belief_state(self, state: State) -> jax.Array:
        """Compute the probability distribution over states from a state vector.

        NOTE: returns nans when state is zeros
        """
        return state * self.normalizing_eigenvector / (state @ self.normalizing_eigenvector)

    @eqx.filter_jit
    def normalize_log_belief_state(self, log_state: jax.Array) -> jax.Array:
        """Compute the log probability distribution over states from a log state vector.

        NOTE: returns nans when log_state is -infs (state is zeros)
        """
        log_prob = log_state + self.log_normalizing_eigenvector
        return log_prob - jax.nn.logsumexp(log_prob)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute the probability distribution of the observations that can be emitted by the process."""
        return (state @ self.transition_matrices @ self.normalizing_eigenvector) / (
            state @ self.normalizing_eigenvector
        )

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_state: State) -> jax.Array:
        """Compute the log probability distribution of the observations that can be emitted by the process."""
        # TODO: fix log math (https://github.com/Astera-org/simplexity/issues/9)
        state = cast(State, jnp.exp(log_state))
        obs_prob_dist = self.observation_probability_distribution(state)
        return jnp.log(obs_prob_dist)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""

        def _scan_fn(state_vector, observation):
            return state_vector @ self.transition_matrices[observation], None

        state_vector, _ = jax.lax.scan(_scan_fn, init=self.state_eigenvector, xs=observations)
        return (state_vector @ self.normalizing_eigenvector) / self._normalizing_constant

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        # TODO: fix log math (https://github.com/Astera-org/simplexity/issues/9)
        prob = self.probability(observations)
        return jnp.log(prob)
