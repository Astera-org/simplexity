"""Independent structure: no conditional dependencies between factors.

Each factor operates independently, always using variant 0.
Joint distribution is the product of independent factor distributions.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class IndependentStructure(eqx.Module):
    """Independent structure with no conditional dependencies.

    Each factor operates independently:
    - All factors always use variant 0
    - No control maps needed
    - Joint distribution: P(t0, t1, ..., tF) = P(t0) * P(t1) * ... * P(tF)

    This is the simplest factored structure.
    """

    def compute_joint_distribution(self, context: ConditionalContext) -> jnp.ndarray:
        """Compute joint distribution as product of independent factors.

        Args:
            context: Conditional context with states and parameters

        Returns:
            Flattened joint distribution of shape [prod(V_i)]
        """
        num_factors = len(context.vocab_sizes)
        states = context.states
        component_types = context.component_types
        transition_matrices = context.transition_matrices
        normalizing_eigenvectors = context.normalizing_eigenvectors

        parts = []
        for i in range(num_factors):
            T_i = transition_matrices[i][0]  # pylint: disable=invalid-name  # T_i is standard notation
            norm_i = normalizing_eigenvectors[i][0] if component_types[i] == "ghmm" else None
            p_i = compute_obs_dist_for_variant(component_types[i], states[i], T_i, norm_i)
            parts.append(p_i)

        joint = parts[0]
        for i in range(1, num_factors):
            joint = (joint[..., None] * parts[i]).reshape(*joint.shape, parts[i].shape[0])

        return joint.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jnp.ndarray, ...],
        context: ConditionalContext,  # pylint: disable=unused-argument  # Required by protocol
    ) -> tuple[jnp.ndarray, ...]:
        """Select variants (always 0 for all factors).

        Args:
            obs_tuple: Tuple of observed tokens (unused)
            context: Conditional context (unused)

        Returns:
            Tuple of variant indices (all zeros)
        """
        return tuple(jnp.array(0, dtype=jnp.int32) for _ in obs_tuple)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters (none for independent structure)."""
        return {}
