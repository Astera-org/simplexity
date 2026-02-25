"""Fully conditional structure: mutual dependencies between all factors.

Each factor's parameter variant is selected based on the tokens of
ALL OTHER factors via a control map, producing mutual dependencies.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant, compute_other_multipliers


class FullyConditional(eqx.Module):
    """Fully conditional structure with mutual dependencies between all factors."""

    control_maps: tuple[jax.Array, ...]
    other_multipliers: tuple[jax.Array, ...]
    other_shapes: tuple[tuple[int, ...], ...]
    perms_py: tuple[tuple[int, ...], ...]
    vocab_sizes_py: tuple[int, ...]
    joint_vocab_size: int

    def __init__(
        self,
        control_maps: tuple[jax.Array, ...],
        vocab_sizes: jax.Array,
    ):
        self.control_maps = tuple(jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes)
        num_factors = len(vocab_sizes)

        import math

        self.joint_vocab_size = math.prod(self.vocab_sizes_py)
        self.other_multipliers = compute_other_multipliers(self.vocab_sizes_py)

        other_shapes: list[tuple[int, ...]] = []
        perms_py: list[tuple[int, ...]] = []
        for i in range(num_factors):
            other_shapes.append(tuple(self.vocab_sizes_py[j] for j in range(num_factors) if j != i))

            others = [j for j in range(num_factors) if j != i]
            axis_pos = {j: pos for pos, j in enumerate(others)}
            perm = []
            for j in range(num_factors):
                if j == i:
                    perm.append(len(others))
                else:
                    perm.append(axis_pos[j])
            perms_py.append(tuple(perm))

        self.other_shapes = tuple(other_shapes)
        self.perms_py = tuple(perms_py)

    def _flatten_other_tokens_index(self, tokens: jax.Array, i: int) -> jax.Array:
        """Flatten other-factor tokens to control map index."""
        mult = self.other_multipliers[i]
        # Multiply elementwise and sum (mult[i] == 0)
        return jnp.sum(tokens * mult)

    def compute_joint_distribution(self, context: ConditionalContext) -> jax.Array:
        """Compute joint distribution using product-of-experts."""
        num_factors = len(context.vocab_sizes)
        states = context.states
        component_types = context.component_types
        transition_matrices = context.transition_matrices
        normalizing_eigenvectors = context.normalizing_eigenvectors
        num_variants = context.num_variants

        # Compute per-factor conditionals
        parts = []
        for i in range(num_factors):
            variant_k = num_variants[i]
            ks = jnp.arange(variant_k, dtype=jnp.int32)

            # Compute all variant distributions for factor i
            def get_dist_i(k: jax.Array, i: int = i) -> jax.Array:
                transition_matrix_k = transition_matrices[i][k]
                norm_k = normalizing_eigenvectors[i][k] if component_types[i] == "ghmm" else None
                return compute_obs_dist_for_variant(component_types[i], states[i], transition_matrix_k, norm_k)

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
        prod_j = parts[0]
        for p in parts[1:]:
            prod_j = prod_j * p

        # Normalize
        sum_j = jnp.sum(prod_j)
        norm_j = jnp.where(sum_j > 0, prod_j / sum_j, jnp.ones_like(prod_j) / self.joint_vocab_size)

        assert isinstance(norm_j, jax.Array)

        return norm_j.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jax.Array, ...],
        context: ConditionalContext,  # pylint: disable=unused-argument
    ) -> tuple[jax.Array, ...]:
        """Select variants based on all other factors' tokens."""
        tokens_arr = jnp.array(obs_tuple)
        variants = []
        for i in range(len(obs_tuple)):
            idx = self._flatten_other_tokens_index(tokens_arr, i)
            k_i = self.control_maps[i][idx]
            variants.append(k_i)
        return tuple(variants)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters for fully conditional structure."""
        return {"control_maps": tuple, "vocab_sizes": jax.Array}
