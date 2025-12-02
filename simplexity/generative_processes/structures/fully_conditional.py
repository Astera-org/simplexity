"""Fully conditional structure: mutual dependencies between all factors.

Each factor's parameter variant is selected based on the tokens of
ALL OTHER factors via a control map, producing mutual dependencies.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class FullyConditional(eqx.Module):
    """Fully conditional structure with mutual dependencies.

    Each factor i selects its variant based on all other factors' tokens.
    Joint distribution uses product-of-experts with normalization.

    Attributes:
        control_maps: Tuple of F arrays. control_maps[i] has shape [prod(V_j for j!=i)]
            mapping flattened other-tokens to variant index for factor i.
        other_multipliers: Precomputed radix multipliers for flattening other tokens
        other_shapes: Reshape targets for conditioning on other factors
        perms_py: Axis permutations to align conditional distributions
        vocab_sizes_py: Python int tuple of vocab sizes for shape operations
        joint_vocab_size: Total vocabulary size (product of all V_i)
    """

    control_maps: tuple[jnp.ndarray, ...]
    other_multipliers: tuple[jnp.ndarray, ...]
    other_shapes: tuple[tuple[int, ...], ...]
    perms_py: tuple[tuple[int, ...], ...]
    vocab_sizes_py: tuple[int, ...]
    joint_vocab_size: int

    def __init__(
        self,
        control_maps: tuple[jnp.ndarray, ...],
        vocab_sizes: jnp.ndarray,
    ):
        """Initialize fully conditional structure.

        Args:
            control_maps: Control maps for each factor. control_maps[i] should
                have shape [prod(V_j for j!=i)] mapping other-factor tokens
                to variant index for factor i.
            vocab_sizes: Array of shape [F] with vocab sizes per factor
        """
        self.control_maps = tuple(jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes)
        F = len(vocab_sizes)

        # Compute joint vocab size
        jv = 1
        for v in self.vocab_sizes_py:
            jv *= v
        self.joint_vocab_size = jv

        # Precompute indexing helpers for each factor
        other_multipliers: list[jnp.ndarray] = []
        other_shapes: list[tuple[int, ...]] = []
        perms_py: list[tuple[int, ...]] = []

        for i in range(F):
            # Compute radix multipliers for "other" factors (excluding i)
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

            # Shape for reshaping conditional [prod_others, V_i] -> [*others, V_i]
            other_shapes.append(tuple(self.vocab_sizes_py[j] for j in range(F) if j != i))

            # Permutation to align [*others, V_i] to [V_0, ..., V_{F-1}]
            others = [j for j in range(F) if j != i]
            axis_pos = {j: pos for pos, j in enumerate(others)}
            perm = []
            for j in range(F):
                if j == i:
                    perm.append(len(others))  # V_i is the last axis
                else:
                    perm.append(axis_pos[j])
            perms_py.append(tuple(perm))

        self.other_multipliers = tuple(other_multipliers)
        self.other_shapes = tuple(other_shapes)
        self.perms_py = tuple(perms_py)

    def _flatten_other_tokens_index(self, tokens: jnp.ndarray, i: int) -> jnp.ndarray:
        """Flatten other-factor tokens to control map index.

        Args:
            tokens: Array of shape [F] with all tokens
            i: Factor index to exclude

        Returns:
            Scalar index for control_maps[i]
        """
        mult = self.other_multipliers[i]
        # Multiply elementwise and sum (mult[i] == 0)
        return jnp.sum(tokens * mult)

    def compute_joint_distribution(self, context: ConditionalContext) -> jnp.ndarray:
        """Compute joint distribution using product-of-experts.

        For each factor i, computes conditional P(t_i | all other t_j),
        then multiplies all conditionals and normalizes.

        Args:
            context: Conditional context with states and parameters

        Returns:
            Flattened joint distribution of shape [prod(V_i)]
        """
        F = len(context.vocab_sizes)
        states = context.states
        component_types = context.component_types
        transition_matrices = context.transition_matrices
        normalizing_eigenvectors = context.normalizing_eigenvectors
        num_variants = context.num_variants

        # Compute per-factor conditionals
        parts = []
        for i in range(F):
            K_i = num_variants[i]
            ks = jnp.arange(K_i, dtype=jnp.int32)

            # Compute all variant distributions for factor i
            def get_dist_i(k: jnp.ndarray) -> jnp.ndarray:
                T_k = transition_matrices[i][k]
                norm_k = normalizing_eigenvectors[i][k] if component_types[i] == "ghmm" else None
                return compute_obs_dist_for_variant(component_types[i], states[i], T_k, norm_k)

            all_pi = jax.vmap(get_dist_i)(ks)  # [K_i, V_i]

            # Select per other-tokens using control map
            cm = self.control_maps[i]  # [prod_others]
            cond = all_pi[cm]  # [prod_others, V_i]

            # Reshape to [*others, V_i]
            cond_nd = cond.reshape(self.other_shapes[i] + (self.vocab_sizes_py[i],))

            # Permute to [V_0, ..., V_{F-1}] with V_i at position i
            aligned = jnp.transpose(cond_nd, self.perms_py[i])
            parts.append(aligned)

        # Product of experts
        J = parts[0]
        for p in parts[1:]:
            J = J * p

        # Normalize
        Z = jnp.sum(J)
        J_norm = jnp.where(Z > 0, J / Z, jnp.ones_like(J) / self.joint_vocab_size)

        assert isinstance(J_norm, jnp.ndarray)

        return J_norm.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jnp.ndarray, ...],
        context: ConditionalContext,
    ) -> tuple[jnp.ndarray, ...]:
        """Select variants based on all other factors' tokens.

        Args:
            obs_tuple: Tuple of observed tokens (one per factor)
            context: Conditional context (unused for fully conditional structure)

        Returns:
            Tuple of variant indices (one per factor)
        """
        tokens_arr = jnp.array(obs_tuple)
        variants = []
        for i in range(len(obs_tuple)):
            idx = self._flatten_other_tokens_index(tokens_arr, i)
            k_i = self.control_maps[i][idx]
            variants.append(k_i)
        return tuple(variants)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters for fully conditional structure."""
        return {"control_maps": tuple, "vocab_sizes": jnp.ndarray}
