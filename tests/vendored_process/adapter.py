"""Project-local adapter making vendored generators modules runnable by simplexity.

This is the worked example the migration guide points at. It is the only file a consumer writes:
the vendored modules stay verbatim, and simplexity is not modified at all.

Two things need bridging. Generators exposes module-level functions over a `NamedTuple` of process
data, so the adapter binds that data once and exposes the operations as methods. And generators'
`generate` returns only the final state, whereas simplexity's training path also wants every
intermediate belief state, so the adapter re-scans rather than calling it.
"""

from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from tests.vendored_process.generators_copy import ghmm_process
from tests.vendored_process.generators_copy.classical import mess


class VendoredGhmmProcess(eqx.Module):
    """A generalized hidden Markov model backed by vendored generators code.

    Structurally satisfies `simplexity.run_management.protocols.GenerativeProcess` without
    importing or subclassing anything from `simplexity.generative_processes`.
    """

    data: ghmm_process.Data

    @property
    def vocab_size(self) -> int:
        """The number of observations that can be emitted by the generative process."""
        return int(self.data.Ts.shape[0])

    @property
    def initial_state(self) -> jax.Array:
        """The stationary belief state that generators' `init` derived from the process."""
        return self.data.eta_0

    def emit_observation(self, state: jax.Array, key: chex.PRNGKey) -> chex.Array:
        """Emit an observation based on the state of the generative process."""
        return ghmm_process.sample(self.data, state, key)

    def transition_states(self, state: jax.Array, obs: chex.Array) -> jax.Array:
        """Evolve the state of the generative process based on the observation."""
        return ghmm_process.update(self.data, state, jnp.asarray(obs))

    def observation_probability_distribution(self, state: jax.Array) -> jax.Array:
        """Compute the distribution over observations that can be emitted from a state."""
        return ghmm_process.obs_dist(self.data, state)

    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute the probability of the process generating a sequence of observations."""
        return ghmm_process.seq_prob(self.data, observations)

    @eqx.filter_vmap(in_axes=(None, 0, 0, None, None))
    def generate(
        self, state: jax.Array, key: chex.PRNGKey, sequence_len: int, return_all_states: bool
    ) -> tuple[Any, chex.Array]:
        """Generate a batch of observation sequences, optionally returning every belief state."""
        keys = jax.random.split(key, sequence_len)

        def step(state: jax.Array, key: chex.PRNGKey) -> tuple[jax.Array, tuple[jax.Array, chex.Array]]:
            obs = self.emit_observation(state, key)
            return self.transition_states(state, obs), (state, obs)

        final_state, (states, observations) = jax.lax.scan(step, state, keys)
        if return_all_states:
            return states, observations
        return final_state, observations


def build_vendored_mess3(x: float, a: float, num_states: int = 3) -> VendoredGhmmProcess:
    """Build a vendored mess3 process.

    Hydra targets this rather than the class, because assembling a process from generators modules
    is code: transition matrices are constructed, passed through `init`, and for composite
    processes bound to consumer-chosen encode/decode callables. None of that fits in YAML.

    Args:
        x: Transition probability to each other state.
        a: Emission probability corresponding to the previous state.
        num_states: Number of hidden states.

    Returns:
        A process satisfying simplexity's generative process protocol.
    """
    return VendoredGhmmProcess(data=ghmm_process.init(mess(x, a, num_states)))
