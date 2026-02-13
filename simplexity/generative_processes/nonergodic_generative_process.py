"""Nonergodic generative process that composes multiple GenerativeProcess components."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.utils.jnp_utils import resolve_jax_device

# Type alias for component states: either a flat array or tuple of arrays (FactoredState)
ComponentState = jax.Array | tuple[jax.Array, ...]


# -----------------------------------------------------------------------------
# Helper functions for handling heterogeneous component state types.
#
# The generate() method uses jax.lax.switch to select which component to use.
# This requires all branches to return the same shape, but components may have
# different state types:
#   - HMM/GHMM: flat jax.Array of shape [state_dim]
#   - FactoredGenerativeProcess: tuple of arrays (FactoredState)
#
# To handle this uniformly, we:
#   1. Flatten each state to a 1D array (preserving total element count)
#   2. Pad to a common max size for switch compatibility
#   3. After processing, unpad and unflatten back to original structure
#
# This lets us work with uniform arrays inside jax.lax.switch while components
# still receive their native state structures.
# -----------------------------------------------------------------------------


def _get_flat_size(state: ComponentState) -> int:
    """Get total number of elements in a component state.

    Args:
        state: Either a flat jax.Array or a tuple of arrays (FactoredState)

    Returns:
        Total element count across all arrays in the state
    """
    if isinstance(state, tuple):
        return sum(arr.size for arr in state)
    return state.size


def _flatten_state(state: ComponentState) -> jax.Array:
    """Flatten a component state to a 1D array.

    Args:
        state: Either a flat jax.Array or a tuple of arrays (FactoredState)

    Returns:
        1D array containing all elements from the state
    """
    if isinstance(state, tuple):
        return jnp.concatenate([arr.ravel() for arr in state])
    return state.ravel()


def _unflatten_state(flat: jax.Array, template: ComponentState) -> ComponentState:
    """Restore original state structure from a flattened 1D array.

    Uses the template to determine:
    - For flat arrays: the target shape
    - For tuples: the number of arrays, each array's shape, and split points

    Args:
        flat: 1D array containing state data
        template: Original state (used only for shape/structure, not values)

    Returns:
        State with same structure as template, populated with data from flat

    Note:
        Uses dynamic_slice instead of split to avoid ConcretizationTypeError
        inside jax.lax.switch. The template shapes are concrete (known at trace
        time), so we can compute offsets as Python ints.
    """
    if isinstance(template, tuple):
        # Extract parts using dynamic_slice with concrete offsets and sizes
        # This avoids jnp.split which requires concrete split indices
        offset = 0
        parts = []
        for t in template:
            part = jax.lax.dynamic_slice(flat, (offset,), (t.size,))
            parts.append(part.reshape(t.shape))
            offset += t.size
        return tuple(parts)
    return flat.reshape(template.shape)


class NonErgodicState(NamedTuple):
    """State for nonergodic generative process.

    Attributes:
        component_beliefs: P(component_i | observations_so_far), shape [num_components].
            Sums to 1. For generation, becomes one-hot after first emission.
        component_states: Tuple of per-component state arrays. Each element has the
            shape expected by that component's GenerativeProcess.
    """

    component_beliefs: jax.Array
    component_states: tuple[Any, ...]


class NonErgodicGenerativeProcess(GenerativeProcess[NonErgodicState]):
    """A nonergodic mixture of generative processes.

    Composes multiple GenerativeProcess instances into a block diagonal structure
    where no transitions occur between components. The process maintains belief
    over which component generated the sequence, updated via Bayes rule.

    Key efficiency: Does NOT materialize a full block diagonal matrix. Instead,
    it stores component processes directly and updates only the relevant beliefs.

    For generation: A single component is sampled at the start of each sequence
    based on component_weights, and all observations come from that component.

    For inference: Beliefs are tracked across all components via Bayesian filtering.

    Attributes:
        components: Tuple of component GenerativeProcess instances.
        component_weights: Initial mixture weights (normalized to sum to 1).
        vocab_maps: Per-component mapping from local vocab to global vocab.
        _vocab_size: Unified vocabulary size across all components.
        _inverse_vocab_maps: Per-component mapping from global vocab to local vocab.
        device: JAX device for arrays.
    """

    components: tuple[GenerativeProcess, ...]
    component_weights: jax.Array
    vocab_maps: tuple[jax.Array, ...]
    _vocab_size: int
    _inverse_vocab_maps: tuple[jax.Array, ...]
    device: jax.Device  # type: ignore[valid-type]

    def __init__(
        self,
        components: Sequence[GenerativeProcess],
        component_weights: jax.Array | Sequence[float],
        vocab_maps: Sequence[Sequence[int]] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize nonergodic generative process.

        Args:
            components: Sequence of GenerativeProcess instances to compose.
            component_weights: Initial mixture weights. Will be normalized to sum to 1.
            vocab_maps: Optional per-component vocab mappings. vocab_maps[i] maps
                component i's local token indices to global token indices.
                If None, assumes all components share the same vocab [0, 1, ..., V-1].
            device: Device to place arrays on (e.g., "cpu", "gpu").

        Raises:
            ValueError: If components is empty or weights don't match component count.
        """
        if len(components) == 0:
            raise ValueError("Must provide at least one component")

        self.device = resolve_jax_device(device)
        self.components = tuple(components)

        # Normalize weights
        weights = jnp.array(component_weights)
        if weights.shape[0] != len(components):
            raise ValueError(
                f"Number of weights ({weights.shape[0]}) must match number of components ({len(components)})"
            )
        if jnp.any(weights < 0):
            raise ValueError("Component weights must be non-negative")
        self.component_weights = weights / jnp.sum(weights)
        self.component_weights = jax.device_put(self.component_weights, self.device)

        # Set up vocab maps
        if vocab_maps is None:
            # Default: each component uses its natural vocab [0, 1, ..., V_i-1]
            vocab_maps = [list(range(c.vocab_size)) for c in components]

        self.vocab_maps = tuple(jax.device_put(jnp.array(vm, dtype=jnp.int32), self.device) for vm in vocab_maps)

        # Compute global vocab size
        self._vocab_size = max(max(vm) for vm in vocab_maps) + 1

        # Build inverse vocab maps (global -> local, -1 if not mapped)
        inverse_maps = []
        for vm in vocab_maps:
            inv = jnp.full((self._vocab_size,), -1, dtype=jnp.int32)
            for local_idx, global_idx in enumerate(vm):
                inv = inv.at[global_idx].set(local_idx)
            inverse_maps.append(jax.device_put(inv, self.device))
        self._inverse_vocab_maps = tuple(inverse_maps)

    @property
    def vocab_size(self) -> int:
        """Unified vocabulary size across all components."""
        return self._vocab_size

    @property
    def initial_state(self) -> NonErgodicState:
        """Initial state with component weights and per-component initial states."""
        return NonErgodicState(
            component_beliefs=self.component_weights,
            component_states=tuple(c.initial_state for c in self.components),
        )

    @eqx.filter_jit
    def observation_probability_distribution(self, state: NonErgodicState) -> jax.Array:
        """Compute P(global_obs | state) as weighted sum over components.

        For each global observation token:
        P(obs | state) = sum_i P(component_i | state) * P(obs | component_i, state_i)

        Where P(obs | component_i, state_i) is computed by:
        1. Getting the probability from component i's distribution
        2. Mapping to global vocab via vocab_map
        3. Returning 0 if the global obs is not in component i's vocab
        """
        global_dist = jnp.zeros(self._vocab_size)

        for i, (component, vm) in enumerate(zip(self.components, self.vocab_maps, strict=False)):
            comp_state = state.component_states[i]
            # Get component's distribution over its local vocab
            local_dist = component.observation_probability_distribution(comp_state)
            # Scatter to global vocab positions
            component_contrib = jnp.zeros(self._vocab_size).at[vm].add(local_dist)
            global_dist = global_dist + state.component_beliefs[i] * component_contrib

        return global_dist

    @eqx.filter_jit
    def log_observation_probability_distribution(self, log_belief_state: NonErgodicState) -> jax.Array:
        """Compute log P(global_obs | state).

        Note: This expects log_belief_state with log-space component_beliefs and component_states.
        """
        log_probs = []

        for i, (component, vm) in enumerate(zip(self.components, self.vocab_maps, strict=False)):
            comp_log_state = log_belief_state.component_states[i]
            comp_log_belief = log_belief_state.component_beliefs[i]

            local_log_dist = component.log_observation_probability_distribution(comp_log_state)
            # Create global distribution with -inf for unmapped tokens
            global_log_dist = jnp.full(self._vocab_size, -jnp.inf)
            global_log_dist = global_log_dist.at[vm].set(local_log_dist)
            # Weight by component belief (in log space: add)
            log_probs.append(comp_log_belief + global_log_dist)

        # Combine via logsumexp across components
        log_probs_stacked = jnp.stack(log_probs, axis=0)
        return jax.nn.logsumexp(log_probs_stacked, axis=0)

    @eqx.filter_jit
    def emit_observation(self, state: NonErgodicState, key: chex.PRNGKey) -> chex.Array:
        """Emit an observation by sampling from the mixture distribution.

        First samples a component based on component_beliefs, then emits from
        that component and maps to global vocab.
        """
        key1, key2 = jax.random.split(key)

        # Sample component based on current beliefs
        component_idx = jax.random.categorical(key1, jnp.log(state.component_beliefs))

        # Emit from chosen component using switch for JIT compatibility
        def emit_from_component(i: int, k: chex.PRNGKey) -> chex.Array:
            comp_state = state.component_states[i]
            local_obs = self.components[i].emit_observation(comp_state, k)
            return self.vocab_maps[i][local_obs]

        global_obs = jax.lax.switch(
            component_idx,
            [partial(emit_from_component, i) for i in range(len(self.components))],
            key2,
        )

        return global_obs

    @eqx.filter_jit
    def transition_states(self, state: NonErgodicState, obs: chex.Array) -> NonErgodicState:
        """Update state given observation using Bayesian filtering.

        1. Compute P(obs | component_i) for each component using current states
        2. Update component_beliefs using Bayes rule
        3. Update each component's internal state independently
        """
        new_component_states = []
        likelihoods = []

        for i, (component, inv_map) in enumerate(zip(self.components, self._inverse_vocab_maps, strict=False)):
            comp_state = state.component_states[i]
            local_obs = inv_map[obs]

            # Get observation probability from this component
            local_dist = component.observation_probability_distribution(comp_state)

            # Likelihood is 0 if obs not in this component's vocab
            likelihood = jnp.where(
                local_obs >= 0,
                local_dist[jnp.clip(local_obs, 0, local_dist.shape[0] - 1)],
                0.0,
            )
            likelihoods.append(likelihood)

            # Update component's internal state
            # Only update if the observation was possible for this component
            # (likelihood > 0 checks both vocab membership AND state feasibility)
            # Bind component via default arg to avoid B023 (late binding in loop)
            new_comp_state = jax.lax.cond(
                likelihood > 0,
                lambda s, lo, c=component: c.transition_states(s, lo),
                lambda s, lo, c=None: s,
                comp_state,
                local_obs,
            )
            new_component_states.append(new_comp_state)

        # Bayes update for component beliefs
        likelihoods_arr = jnp.array(likelihoods)
        unnorm_beliefs = state.component_beliefs * likelihoods_arr
        # Handle case where all likelihoods are 0 (shouldn't happen with valid obs)
        normalizer = jnp.sum(unnorm_beliefs)
        new_beliefs = jnp.where(
            normalizer > 0,
            unnorm_beliefs / normalizer,
            state.component_beliefs,  # Keep old beliefs if normalizer is 0
        )

        return NonErgodicState(
            component_beliefs=new_beliefs,
            component_states=tuple(new_component_states),
        )

    @eqx.filter_jit
    def probability(self, observations: jax.Array) -> jax.Array:
        """Compute P(observations) by marginalizing over components.

        P(obs_1:T) = sum_i P(component_i) * P(obs_1:T | component_i)
        """

        def compute_component_prob(i: int) -> jax.Array:
            component = self.components[i]
            inv_map = self._inverse_vocab_maps[i]
            # Map global observations to local
            local_obs = inv_map[observations]
            # Check if all observations are valid for this component
            all_valid = jnp.all(local_obs >= 0)
            # Compute probability (0 if any obs invalid)
            prob = jax.lax.cond(
                all_valid,
                lambda lo: component.probability(lo),
                lambda lo: jnp.array(0.0),
                local_obs,
            )
            return self.component_weights[i] * prob

        # Sum over components
        total_prob = jnp.array(0.0)
        for i in range(len(self.components)):
            total_prob = total_prob + compute_component_prob(i)

        return total_prob

    @eqx.filter_jit
    def log_probability(self, observations: jax.Array) -> jax.Array:
        """Compute log P(observations) using logsumexp for numerical stability."""

        def compute_component_log_prob(i: int) -> jax.Array:
            component = self.components[i]
            inv_map = self._inverse_vocab_maps[i]
            # Map global observations to local
            local_obs = inv_map[observations]
            # Check if all observations are valid for this component
            all_valid = jnp.all(local_obs >= 0)
            # Compute log probability (-inf if any obs invalid)
            log_prob = jax.lax.cond(
                all_valid,
                lambda lo: component.log_probability(lo),
                lambda lo: jnp.array(-jnp.inf),
                local_obs,
            )
            return jnp.log(self.component_weights[i]) + log_prob

        # Combine via logsumexp
        log_probs = jnp.array([compute_component_log_prob(i) for i in range(len(self.components))])
        return jax.nn.logsumexp(log_probs)

    @eqx.filter_vmap(in_axes=(None, 0, 0, None, None))
    def generate(
        self,
        state: NonErgodicState,
        key: chex.PRNGKey,
        sequence_len: int,
        return_all_states: bool,
    ) -> tuple[NonErgodicState, chex.Array]:
        """Generate a sequence from a single sampled component.

        Unlike inference (which tracks beliefs across all components), generation
        samples ONE component at the start and generates entirely from that component.

        This method is vmapped, so inside the function body we work with unbatched
        (single-element) states and keys. We cannot call component.generate() here
        because that method is also vmapped and expects batched inputs. Instead,
        we implement generation directly using jax.lax.scan over emit_observation
        and transition_states.

        Args:
            state: Initial NonErgodicState with component_beliefs and component_states.
                The batch dimension is handled by vmap.
            key: Random key for this sequence.
            sequence_len: Length of sequence to generate.
            return_all_states: If True, return state trajectory at each timestep.

        Returns:
            Tuple of (final_state or state_trajectory, observations).
            States are NonErgodicState. Observations are in global vocab space.
        """
        key1, key2 = jax.random.split(key)
        keys = jax.random.split(key2, sequence_len)

        # 1. Sample which component to use for this entire sequence
        component_idx = jax.random.categorical(key1, jnp.log(state.component_beliefs))

        # 2. Flatten and pad component states for jax.lax.switch compatibility
        #
        # jax.lax.switch requires all branches to return arrays of identical shape.
        # But components can have different state structures:
        #   - HMM/GHMM: flat array of shape [state_dim]
        #   - FactoredGenerativeProcess: tuple of arrays (FactoredState)
        #
        # Our approach:
        #   a) Save original state structures as templates (for unflattening later)
        #   b) Flatten each state to 1D (handles both arrays and tuples)
        #   c) Pad to a common max size
        #   d) Inside switch: unpad + unflatten → call component → flatten + repad
        #   e) After scan: unpad + unflatten to restore native structures
        #
        # Components never see the padding - they work with their native shapes.
        num_components = len(self.components)

        # Store original structures as templates for restoring after processing
        state_templates = state.component_states
        flat_sizes = [_get_flat_size(s) for s in state_templates]
        max_flat_size = max(flat_sizes)

        def flatten_and_pad(s: ComponentState) -> jax.Array:
            """Convert any state to padded 1D array for uniform shape in switch."""
            flat = _flatten_state(s)
            return jnp.pad(flat, (0, max_flat_size - flat.size))

        def unpad_and_unflatten(padded: jax.Array, original_size: int, template: ComponentState) -> ComponentState:
            """Restore original state structure from padded 1D array."""
            flat = padded[:original_size]
            return _unflatten_state(flat, template)

        padded_states = tuple(flatten_and_pad(s) for s in state.component_states)

        # 3. Generate sequence using scan, selecting component via switch at each step
        def gen_step_for_component(
            i: int, padded_state: jax.Array, step_key: chex.PRNGKey
        ) -> tuple[jax.Array, chex.Array]:
            """Generate one observation from component i and update its state."""
            # Unpad and unflatten to get native state structure
            real_state = unpad_and_unflatten(padded_state, flat_sizes[i], state_templates[i])
            # Call component's methods with native state shapes
            local_obs = self.components[i].emit_observation(real_state, step_key)
            new_real_state = self.components[i].transition_states(real_state, local_obs)
            # Flatten and repad for uniform shape in switch
            new_padded_state = flatten_and_pad(new_real_state)
            global_obs = self.vocab_maps[i][local_obs]
            return new_padded_state, global_obs

        def scan_step(
            carry: tuple[jax.Array, tuple[jax.Array, ...]], step_key: chex.PRNGKey
        ) -> tuple[tuple[jax.Array, tuple[jax.Array, ...]], chex.Array]:
            """One generation step: emit from active component, update its state."""
            idx, padded_comp_states = carry

            # Use switch to call the correct component's emit/transition
            def gen_from_i(i: int) -> tuple[jax.Array, chex.Array]:
                return gen_step_for_component(i, padded_comp_states[i], step_key)

            new_padded_state, global_obs = jax.lax.switch(
                idx,
                [partial(gen_from_i, i) for i in range(num_components)],
            )

            # Update only the active component's state, keep others unchanged
            # Now all states have uniform shape, so cond works
            new_padded_comp_states = tuple(
                jax.lax.cond(
                    idx == i,
                    lambda ns=new_padded_state: ns,
                    lambda ps=padded_comp_states[i]: ps,
                )
                for i in range(num_components)
            )

            return (idx, new_padded_comp_states), global_obs

        # Run the scan over all timesteps with padded states
        init_carry = (component_idx, padded_states)
        (_, final_padded_states), observations = jax.lax.scan(scan_step, init_carry, keys)

        # 4. Unpad and unflatten final states back to native structures
        final_comp_states = tuple(
            unpad_and_unflatten(final_padded_states[i], flat_sizes[i], state_templates[i])
            for i in range(num_components)
        )

        # 5. Construct final state with one-hot beliefs
        one_hot_beliefs = jax.nn.one_hot(component_idx, len(self.components), dtype=self.component_weights.dtype)

        if return_all_states:
            # Run inference pass to compute state trajectory
            # This simulates what an observer would compute via Bayesian filtering
            # on the generated sequence, starting from the original initial state.
            def inference_step(
                carry_state: NonErgodicState, obs: chex.Array
            ) -> tuple[NonErgodicState, NonErgodicState]:
                new_state = self.transition_states(carry_state, obs)
                return new_state, carry_state

            # Use original initial state for inference (not modified by generation)
            _, state_trajectory = jax.lax.scan(inference_step, state, observations)

            return state_trajectory, observations
        else:
            return NonErgodicState(
                component_beliefs=one_hot_beliefs,
                component_states=final_comp_states,
            ), observations
