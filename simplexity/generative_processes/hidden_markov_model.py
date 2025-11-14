"""Hidden Markov Model class."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from typing import cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel, State
from simplexity.generative_processes.transition_matrices import get_stationary_state


class HiddenMarkovModel(GeneralizedHiddenMarkovModel[State]):
    """A Hidden Markov Model."""

    def __init__(self, transition_matrices: jax.Array, initial_state: jax.Array | None = None):
        self.validate_transition_matrices(transition_matrices)

        state_transition_matrix = jnp.sum(transition_matrices, axis=0)
        eigenvalues, _ = jnp.linalg.eig(state_transition_matrix)
        principal_eigenvalue = jnp.max(eigenvalues)

        if jnp.isclose(principal_eigenvalue, 1):
            self.transition_matrices = transition_matrices
        else:
            self.transition_matrices = transition_matrices / principal_eigenvalue
        self.log_transition_matrices = jnp.log(transition_matrices)

        self.normalizing_eigenvector = jnp.ones(self.num_states)
        self.log_normalizing_eigenvector = jnp.zeros(self.num_states)

        if initial_state is None:
            initial_state = get_stationary_state(state_transition_matrix.T)
        self._initial_state = initial_state
        self.log_initial_state = jnp.log(self._initial_state)

        self.normalizing_constant = jnp.sum(self._initial_state)
        self.log_normalizing_constant = jax.nn.logsumexp(self.log_initial_state)

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
    def normalize_log_belief_state(self, log_belief_state: jax.Array) -> jax.Array:
        """Compute the log probability distribution over states from a log state vector."""
        return log_belief_state - jax.nn.logsumexp(log_belief_state)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute the probability distribution of the observations that can be emitted by the process."""
        obs_state_dist = state @ self.transition_matrices
        return jnp.sum(obs_state_dist, axis=1)

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: State) -> jax.Array:
        """Compute the log probability distribution of the observations that can be emitted by the process."""
        log_obs_state_dist = jax.nn.logsumexp(log_belief_state[:, None] + self.log_transition_matrices, axis=1)
        return jax.nn.logsumexp(log_obs_state_dist, axis=1)

    @eqx.filter_jit
    def probability(self, observations: jax.Array, return_all_probabilities: bool = False) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        if return_all_probabilities:
            def _scan_fn_cummulative(state_vector, observation):
                new_state = state_vector @ self.transition_matrices[observation]
                prob = jnp.sum(new_state) / self.normalizing_constant
                return new_state, prob

            _, probs = jax.lax.scan(_scan_fn_cummulative, init=self._initial_state, xs=observations)
            return probs
        else:
            def _scan_fn(state_vector, observation):
                return state_vector @ self.transition_matrices[observation], None

            state_vector, _ = jax.lax.scan(_scan_fn, init=self._initial_state, xs=observations)
            return jnp.sum(state_vector) / self.normalizing_constant

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array, return_all_probabilities: bool = False) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        if return_all_probabilities:
            def _scan_fn_cumulative(log_belief_state, observation):
                new_log_state = jax.nn.logsumexp(log_belief_state[:, None] + self.log_transition_matrices[observation], axis=0)
                log_prob = jax.nn.logsumexp(new_log_state) - self.log_normalizing_constant
                return new_log_state, log_prob

            _, log_probs = jax.lax.scan(_scan_fn_cumulative, init=self.log_initial_state, xs=observations)
            return log_probs
        else:
            def _scan_fn(log_belief_state, observation):
                return jax.nn.logsumexp(log_belief_state[:, None] + self.log_transition_matrices[observation], axis=0), None

            log_belief_state, _ = jax.lax.scan(_scan_fn, init=self.log_initial_state, xs=observations)
            return jax.nn.logsumexp(log_belief_state) - self.log_normalizing_constant
