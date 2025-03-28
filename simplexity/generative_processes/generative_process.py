from abc import abstractmethod
from typing import Generic, TypeVar

import chex
import equinox as eqx
import jax

State = TypeVar("State")


class GenerativeProcess(eqx.Module, Generic[State]):
    """A generative process is a probabilistic model that can be used to generate data."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """The number of observations that can be emitted by the generative process."""
        ...

    @property
    @abstractmethod
    def initial_state(self) -> State:
        """The initial state of the generative process."""
        ...

    @abstractmethod
    def emit_observation(self, state: State, key: chex.PRNGKey) -> chex.Array:
        """Emit an observation based on the state of the generative process."""
        ...

    @abstractmethod
    def transition_states(self, state: State, obs: chex.Array) -> State:
        """Evolve the state of the generative process based on the observation.

        The input state represents a prior distribution over hidden states, and
        the returned state represents a posterior distribution over hidden states
        conditioned on the observation.
        """
        ...

    @eqx.filter_vmap(in_axes=(None, 0, 0, None))
    def generate(self, state: State, key: chex.PRNGKey, sequence_len: int) -> tuple[State, chex.Array]:
        """Generate a batch of sequences of observations from the generative process.

        Returns:
            A tuple of (final_states, observations) where:
            - observations has shape (batch_size, sequence_len, observation_dim)
            - final_states has shape (batch_size,) + state_shape
        """
        keys = jax.random.split(key, sequence_len)

        def scan_fn(carry: State, key: chex.PRNGKey) -> tuple[State, chex.Array]:
            obs = self.emit_observation(carry, key)
            carry = self.transition_states(carry, obs)
            return carry, obs

        return jax.lax.scan(scan_fn, state, keys)

    @eqx.filter_vmap(in_axes=(None, 0, 0, None))
    def generate_full(
        self, state: State, key: chex.PRNGKey, sequence_len: int
    ) -> tuple[chex.Array, chex.Array, chex.PRNGKey]:
        """Generate a batch of sequences of hidden states and observations from the generative process.

        Returns:
            A tuple of (belief_states, observations, keys) where:
            - belief_states has shape (batch_size, sequence_len, num_states)
            - observations has shape (batch_size, sequence_len)
            - keys has shape (batch_size, 2) (the resulting successor keys)
        """

        # scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
        def scan_fn(carry: tuple[State, chex.PRNGKey], _x) -> tuple[tuple[State, chex.PRNGKey], chex.Array]:
            state, key = carry
            obs = self.emit_observation(state, key)
            new_state = self.transition_states(state, obs)
            key, _ = jax.random.split(key)
            carry = new_state, key
            return carry, (state, obs)

        (state, key), (states, obs) = jax.lax.scan(scan_fn, (state, key), length=sequence_len)
        return key, states, obs

    @abstractmethod
    def observation_probability_distribution(self, state: State) -> jax.Array:
        """Compute the probability distribution of the observations that can be emitted by the process."""
        ...

    @abstractmethod
    def log_observation_probability_distribution(self, log_belief_state: State) -> jax.Array:
        """Compute the log probability distribution of the observations that can be emitted by the process."""
        ...

    @abstractmethod
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        ...

    @abstractmethod
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute the log probability of the process generating a sequence of observations."""
        ...
