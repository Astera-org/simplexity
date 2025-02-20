from typing import cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel, State


class HiddenMarkovModel(GeneralizedHiddenMarkovModel[State]):
    """A Hidden Markov Model."""

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

        self.normalizing_eigenvector = jnp.ones(self.num_states)
        self.log_normalizing_eigenvector = jnp.zeros(self.num_states)

        eigenvalues, left_eigenvectors = jnp.linalg.eig(state_transition_matrix.T)
        state_eigenvector = left_eigenvectors[:, jnp.isclose(eigenvalues, principal_eigenvalue)].squeeze().real
        self.state_eigenvector = state_eigenvector / jnp.sum(state_eigenvector)
        self.log_state_eigenvector = jnp.log(self.state_eigenvector)

        self._normalizing_constant = jnp.sum(self.state_eigenvector)
        self._log_normalizing_constant = jax.nn.logsumexp(self.log_state_eigenvector)

    def validate_transition_matrices(self, transition_matrices: jax.Array):
        """Validate the transition matrices."""
        super().validate_transition_matrices(transition_matrices)
        assert jnp.all(transition_matrices >= 0)
        assert jnp.all(transition_matrices <= 1)
        sum_over_obs_and_next = jnp.sum(transition_matrices, axis=(0, 2))
        chex.assert_trees_all_close(sum_over_obs_and_next, jnp.ones_like(sum_over_obs_and_next))

    @eqx.filter_jit
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Evolve the state of the generative process based on the observation.

        The input state represents a prior distribution over hidden states, and
        the returned state represents a posterior distribution over hidden states
        conditioned on the observation.
        """
        state = cast(State, state @ self.transition_matrices[obs])
        return cast(State, self.normalize_belief_state(state))

    @eqx.filter_jit
    def normalize_belief_state(self, state: State) -> jax.Array:
        """Compute the probability distribution over states from a state vector."""
        return state / jnp.sum(state)

    @eqx.filter_jit
    def normalize_log_belief_state(self, log_state: jax.Array) -> jax.Array:
        """Compute the log probability distribution over states from a log state vector."""
        return log_state - jax.nn.logsumexp(log_state)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute the probability distribution of the observations that can be emitted by the generative process."""
        return jnp.sum(state @ self.transition_matrices, axis=1)

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""

        def _scan_fn(state_vector, observation):
            return state_vector @ self.transition_matrices[observation], None

        state_vector, _ = jax.lax.scan(_scan_fn, init=self.state_eigenvector, xs=observations)
        return jnp.sum(state_vector) / self._normalizing_constant

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""

        def _scan_fn(log_state_vector, observation):
            return jax.nn.logsumexp(log_state_vector[:, None] + self.log_transition_matrices[observation], axis=0), None

        log_state_vector, _ = jax.lax.scan(_scan_fn, init=self.log_state_eigenvector, xs=observations)
        return jax.nn.logsumexp(log_state_vector) - self._log_normalizing_constant
