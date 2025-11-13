"""Transition-coupled topology: independent/chain emissions with coupled transitions.

Emissions can be either:
- Independent: each factor uses a fixed emission variant
- Chain-style: factor i selects emission variant based on previous tokens 0..i-1

Transitions are always coupled: factor i selects transition variant based on
all other factors' tokens.
"""

from __future__ import annotations
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.factored_beliefs.topology.topology import CouplingContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class TransitionCoupledTopology(eqx.Module):
    """Transition-coupled topology with flexible emission modes.

    Emissions can be:
    - Independent (use_emission_chain=False): P(t) = ∏_i P_i(t_i | s_i, k_emit_i)
    - Chain (use_emission_chain=True): P(t) = P0(t0) * ∏_{i>0} P_i(t_i | t_0..t_{i-1}, s_i)

    Transitions are always coupled: factor i selects transition variant based on
    all other factors' tokens.

    Attributes:
        control_maps_transition: Transition control maps. control_maps_transition[i]
            has shape [prod(V_j for j!=i)] mapping other tokens to transition variant.
        emission_variant_indices: Fixed emission variants per factor (shape [F])
        emission_control_maps: Optional chain-style emission control maps
        use_emission_chain: Whether to use chain-style emissions
        other_multipliers: Precomputed radix multipliers for other-factor indexing
        prefix_multipliers: Precomputed radix multipliers for prefix indexing
        vocab_sizes_py: Python int tuple of vocab sizes
    """

    control_maps_transition: tuple[jnp.ndarray, ...]
    emission_variant_indices: jnp.ndarray  # shape [F]
    emission_control_maps: tuple[jnp.ndarray | None, ...]
    use_emission_chain: bool
    other_multipliers: tuple[jnp.ndarray, ...]
    prefix_multipliers: tuple[jnp.ndarray, ...]
    vocab_sizes_py: tuple[int, ...]

    def __init__(
        self,
        control_maps_transition: tuple[jnp.ndarray, ...],
        emission_variant_indices: jnp.ndarray | Sequence[int],
        vocab_sizes: jnp.ndarray,
        emission_control_maps: tuple[jnp.ndarray | None, ...] | None = None,
    ):
        """Initialize transition-coupled topology.

        Args:
            control_maps_transition: Transition control maps for each factor.
                control_maps_transition[i] should have shape [prod(V_j for j!=i)].
            emission_variant_indices: Fixed emission variant per factor (shape [F])
            vocab_sizes: Vocabulary sizes per factor (shape [F])
            emission_control_maps: Optional chain-style emission control maps.
                If provided, emission_control_maps[i] should have shape
                [prod(V_j for j<i)] for i>0.
        """
        self.control_maps_transition = tuple(
            jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps_transition
        )
        self.emission_variant_indices = jnp.asarray(emission_variant_indices, dtype=jnp.int32)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes)
        F = len(vocab_sizes)

        # Process emission control maps
        use_chain = False
        ecm_list: list[jnp.ndarray | None] = []
        if emission_control_maps is not None:
            for i, cm_i in enumerate(emission_control_maps):
                if cm_i is None:
                    ecm_list.append(None)
                else:
                    ecm_list.append(jnp.asarray(cm_i, dtype=jnp.int32))
                    if i > 0:
                        use_chain = True
        else:
            ecm_list = [None] * F
        self.emission_control_maps = tuple(ecm_list)
        self.use_emission_chain = bool(use_chain)

        # Precompute multipliers for other-factor indexing (for transitions)
        other_multipliers: list[jnp.ndarray] = []
        for i in range(F):
            mult = []
            for j in range(F):
                if j == i:
                    mult.append(0)  # Unused
                else:
                    m = 1
                    for k in range(j + 1, F):
                        if k == i:
                            continue
                        m *= self.vocab_sizes_py[k]
                    mult.append(m)
            other_multipliers.append(jnp.array(mult))
        self.other_multipliers = tuple(other_multipliers)

        # Precompute multipliers for prefix indexing (for chain emissions)
        prefix_multipliers: list[jnp.ndarray] = []
        for i in range(F):
            pmult = []
            for j in range(F):
                if j >= i:
                    pmult.append(0)  # Unused
                else:
                    m = 1
                    for k in range(j + 1, i):
                        m *= self.vocab_sizes_py[k]
                    pmult.append(m)
            prefix_multipliers.append(jnp.array(pmult))
        self.prefix_multipliers = tuple(prefix_multipliers)

    def _flatten_other_tokens_index(self, tokens: jnp.ndarray, i: int) -> jnp.ndarray:
        """Flatten other-factor tokens to transition control map index."""
        mult = self.other_multipliers[i]
        return jnp.sum(tokens * mult)

    def _flatten_prev_tokens_index(self, tokens: jnp.ndarray, i: int) -> jnp.ndarray:
        """Flatten prefix tokens to emission control map index."""
        mult = self.prefix_multipliers[i]
        return jnp.sum(tokens * mult)

    def compute_joint_distribution(self, context: CouplingContext) -> jnp.ndarray:
        """Compute joint distribution based on emission mode.

        Args:
            context: Coupling context with states and parameters

        Returns:
            Flattened joint distribution of shape [prod(V_i)]
        """
        F = len(context.vocab_sizes)
        states = context.states
        component_types = context.component_types
        transition_matrices = context.transition_matrices
        normalizing_eigenvectors = context.normalizing_eigenvectors
        num_variants = context.num_variants

        if not self.use_emission_chain:
            # Independent emissions
            parts = []
            for i in range(F):
                k_emit = self.emission_variant_indices[i]
                T_k = transition_matrices[i][k_emit]
                norm_k = normalizing_eigenvectors[i][k_emit] if component_types[i] == "ghmm" else None
                p_i = compute_obs_dist_for_variant(component_types[i], states[i], T_k, norm_k)
                parts.append(p_i)

            # Product of independent factors
            J = parts[0]
            for i in range(1, F):
                J = (J[..., None] * parts[i]).reshape(*J.shape, parts[i].shape[0])
            return J.reshape(-1)

        # Chain-style emissions
        k0 = self.emission_variant_indices[0]
        T0 = transition_matrices[0][k0]
        norm0 = normalizing_eigenvectors[0][k0] if component_types[0] == "ghmm" else None
        joint = compute_obs_dist_for_variant(component_types[0], states[0], T0, norm0)
        prev_prod = self.vocab_sizes_py[0]

        for i in range(1, F):
            K_i = num_variants[i]
            ks = jnp.arange(K_i, dtype=jnp.int32)

            # Compute all variant distributions
            def get_dist_i(k: jnp.ndarray) -> jnp.ndarray:
                T_k = transition_matrices[i][k]
                norm_k = normalizing_eigenvectors[i][k] if component_types[i] == "ghmm" else None
                return compute_obs_dist_for_variant(component_types[i], states[i], T_k, norm_k)

            all_pi = jax.vmap(get_dist_i)(ks)  # [K_i, V_i]

            cm = self.emission_control_maps[i]
            if cm is None:
                # Use fixed emission variant
                fixed = self.emission_variant_indices[i]
                cond = jnp.tile(all_pi[fixed][None, :], (prev_prod, 1))  # [prev_prod, V_i]
            else:
                # Use control map
                cond = all_pi[cm]  # [prev_prod, V_i]

            left = joint.reshape(prev_prod)
            extended = cond * left[:, None]
            curr_vocab_size = self.vocab_sizes_py[i]
            joint = extended.reshape(*(list(joint.shape) + [curr_vocab_size]))
            prev_prod *= curr_vocab_size

        return joint.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jnp.ndarray, ...],
        context: CouplingContext,
    ) -> tuple[jnp.ndarray, ...]:
        """Select transition variants based on other factors' tokens.

        Note: This returns TRANSITION variants, not emission variants.

        Args:
            obs_tuple: Tuple of observed tokens (one per factor)
            context: Coupling context (unused)

        Returns:
            Tuple of transition variant indices (one per factor)
        """
        tokens_arr = jnp.array(obs_tuple)
        variants = []
        for i in range(len(obs_tuple)):
            idx = self._flatten_other_tokens_index(tokens_arr, i)
            k_trans = self.control_maps_transition[i][idx]
            variants.append(k_trans)
        return tuple(variants)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters for transition-coupled topology."""
        return {
            "control_maps_transition": tuple,
            "emission_variant_indices": jnp.ndarray,
            "vocab_sizes": jnp.ndarray,
            "emission_control_maps": tuple,  # optional
        }
