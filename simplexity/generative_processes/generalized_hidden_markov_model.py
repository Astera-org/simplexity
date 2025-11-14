"""Generalized Hidden Markov Model class."""

# pylint: disable-all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

from typing import TypeVar, cast

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.generative_processes.transition_matrices import get_stationary_state

State = TypeVar("State", bound=jax.Array)


class GeneralizedHiddenMarkovModel(GenerativeProcess[State]):
    """A Generalized Hidden Markov Model."""

    transition_matrices: jax.Array
    log_transition_matrices: jax.Array
    normalizing_eigenvector: jax.Array
    log_normalizing_eigenvector: jax.Array
    _initial_state: jax.Array
    log_initial_state: jax.Array
    normalizing_constant: jax.Array
    log_normalizing_constant: jax.Array

    def __init__(self, transition_matrices: jax.Array, initial_state: jax.Array | None = None):
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

        if initial_state is None:
            initial_state = get_stationary_state(state_transition_matrix.T)
        self._initial_state = initial_state
        self.log_initial_state = jnp.log(self._initial_state)

        self.normalizing_constant = self._initial_state @ self.normalizing_eigenvector
        self.log_normalizing_constant = jax.nn.logsumexp(self.log_initial_state + self.log_normalizing_eigenvector)

    def validate_transition_matrices(self, transition_matrices: jax.Array):
        """Validate the transition matrices."""
        if transition_matrices.ndim != 3 or transition_matrices.shape[1] != transition_matrices.shape[2]:
            raise ValueError("Transition matrices must have shape (vocab_size, num_states, num_states)")

    @property
    def vocab_size(self) -> int:
        """The number of distinct observations that can be emitted by the model."""
        return self.transition_matrices.shape[0]

    @property
    def num_states(self) -> int:
        """The number of hidden states in the model."""
        return self.transition_matrices.shape[1]

    @property
    def initial_state(self) -> State:
        """The initial state of the model."""
        return cast(State, self._initial_state)

    @eqx.filter_jit
    def emit_observation(self, state: State, key: chex.PRNGKey) -> jax.Array:
        """Emit an observation based on the state of the generative process."""
        obs_probs = self.observation_probability_distribution(state)
        return jax.random.choice(key, self.vocab_size, p=obs_probs)

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
        return state / (state @ self.normalizing_eigenvector)

    @eqx.filter_jit
    def normalize_log_belief_state(self, log_belief_state: jax.Array) -> jax.Array:
        """Compute the log probability distribution over states from a log state vector.

        NOTE: returns nans when log_belief_state is -infs (state is zeros)
        """
        return log_belief_state - jax.nn.logsumexp(log_belief_state + self.log_normalizing_eigenvector)

    @eqx.filter_jit
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute the probability distribution of the observations that can be emitted by the process."""
        return (state @ self.transition_matrices @ self.normalizing_eigenvector) / (
            state @ self.normalizing_eigenvector
        )

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: State) -> jax.Array:
        """Compute the log probability distribution of the observations that can be emitted by the process."""
        # TODO: fix log math (https://github.com/Astera-org/simplexity/issues/9)
        state = cast(State, jnp.exp(log_belief_state))
        obs_prob_dist = self.observation_probability_distribution(state)
        return jnp.log(obs_prob_dist)

    @eqx.filter_jit
    def probability(self, observations: jax.Array, return_all_probabilities: bool = False) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        if return_all_probabilities:
            def _scan_fn_cumulative(state_vector, observation):
                new_state = state_vector @ self.transition_matrices[observation]
                prob = (new_state @ self.normalizing_eigenvector) / self.normalizing_constant
                return new_state, prob

            _, probs = jax.lax.scan(_scan_fn_cumulative, init=self._initial_state, xs=observations)
            return probs
        else:
            def _scan_fn(state_vector, observation):
                return state_vector @ self.transition_matrices[observation], None

            state_vector, _ = jax.lax.scan(_scan_fn, init=self._initial_state, xs=observations)
            return (state_vector @ self.normalizing_eigenvector) / self.normalizing_constant

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array, return_all_probabilities: bool = False) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        # TODO: fix log math (https://github.com/Astera-org/simplexity/issues/9)
        prob = self.probability(observations, return_all_probabilities=return_all_probabilities)
        return jnp.log(prob)
