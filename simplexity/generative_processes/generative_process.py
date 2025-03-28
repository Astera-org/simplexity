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

    @eqx.filter_vmap(in_axes=(None, 0, 0, None, None))
    def generate(
        self, state: State, key: chex.PRNGKey, sequence_len: int, return_intermediate_states: bool = False
    ) -> tuple[State, chex.Array]:
        """Generate a batch of sequences of observations from the generative process."""
        keys = jax.random.split(key, sequence_len)

        def gen_obs(state: State, key: chex.PRNGKey) -> tuple[State, chex.Array]:
            obs = self.emit_observation(state, key)
            state = self.transition_states(state, obs)
            return state, obs

        def gen_states_and_obs(state: State, key: chex.PRNGKey) -> tuple[State, tuple[State, chex.Array]]:
            obs = self.emit_observation(state, key)
            state = self.transition_states(state, obs)
            return state, (state, obs)

        if return_intermediate_states:
            _, (states, obs) = jax.lax.scan(gen_states_and_obs, state, keys)
            return states, obs

        return jax.lax.scan(gen_obs, state, keys)

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
