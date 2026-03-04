"""Fully conditional structure: mutual dependencies between all factors.

Each factor's parameter variant is selected based on the tokens of
ALL OTHER factors via a control map, producing mutual dependencies.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.structures.indexing import build_other_factor_multipliers, flatten_index
from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class FullyConditional(eqx.Module):
    """Fully conditional structure with mutual dependencies.

    Each factor i selects its variant based on all other factors' tokens.
    Joint distribution uses a normalized product-of-conditionals approximation.

    Attributes:
        control_maps: Tuple of F arrays. control_maps[i] has shape [prod(V_j for j!=i)]
            mapping flattened other-tokens to variant index for factor i.
        other_multipliers: Precomputed radix multipliers for flattening other tokens
        other_shapes: Reshape targets for conditioning on other factors
        perms_py: Axis permutations to align conditional distributions
        vocab_sizes_py: Python int tuple of vocab sizes for shape operations
        joint_vocab_size: Total vocabulary size (product of all V_i)
    """

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
        """Initialize fully conditional structure.

        Args:
            control_maps: Control maps for each factor. control_maps[i] should
                have shape [prod(V_j for j!=i)] mapping other-factor tokens
                to variant index for factor i.
            vocab_sizes: Array of shape [F] with vocab sizes per factor
        """
        self.control_maps = tuple(jnp.asarray(cm, dtype=jnp.int32) for cm in control_maps)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes)
        num_factors = len(self.vocab_sizes_py)

        if num_factors == 0:
            raise ValueError("FullyConditional requires at least one factor")
        if len(self.control_maps) != num_factors:
            raise ValueError(f"Expected {num_factors} control maps (one per factor), got {len(self.control_maps)}")
        if any(v <= 0 for v in self.vocab_sizes_py):
            raise ValueError(f"All vocab sizes must be positive, got {self.vocab_sizes_py}")

        # Compute joint vocab size
        jv = 1
        for v in self.vocab_sizes_py:
            jv *= v
        self.joint_vocab_size = jv

        # Precompute indexing helpers for each factor
        other_shapes: list[tuple[int, ...]] = []
        perms_py: list[tuple[int, ...]] = []

        # Validate control maps and build shape/alignment metadata
        for i in range(num_factors):
            cm = self.control_maps[i]
            if cm.ndim != 1:
                raise ValueError(f"control_maps[{i}] must be 1D, got shape {cm.shape}")
            expected_len = 1
            for j, vj in enumerate(self.vocab_sizes_py):
                if j != i:
                    expected_len *= vj
            if int(cm.shape[0]) != expected_len:
                raise ValueError(
                    f"control_maps[{i}] length {cm.shape[0]} must equal prod(V_j for j!=[{i}]) = {expected_len}"
                )

            # Shape for reshaping conditional [prod_others, V_i] -> [*others, V_i]
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

        self.other_multipliers = build_other_factor_multipliers(self.vocab_sizes_py)
        self.other_shapes = tuple(other_shapes)
        self.perms_py = tuple(perms_py)

    def _flatten_other_tokens_index(self, tokens: jax.Array, i: int) -> jax.Array:
        """Flatten other-factor tokens to control map index."""
        mult = self.other_multipliers[i]
        return flatten_index(tokens, mult)

    def compute_joint_distribution(self, context: ConditionalContext) -> jax.Array:
        """Compute an approximate joint distribution via product-of-conditionals.

        For each factor i, computes conditional P(t_i | all other t_j),
        then multiplies all conditionals and normalizes.

        Args:
            context: Conditional context with states and parameters

        Returns:
            Flattened approximate joint distribution of shape [prod(V_i)]
        """
        num_factors = len(context.vocab_sizes)
        states = context.states
        component_types = context.component_types
        transition_matrices = context.transition_matrices
        normalizing_eigenvectors = context.normalizing_eigenvectors
        num_variants = context.num_variants

        # Compute per-factor conditional log probabilities aligned to [V_0, ..., V_{F-1}]
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
            aligned_log = jnp.where(aligned > 0.0, jnp.log(aligned), -jnp.inf)
            parts.append(aligned_log)

        # Product in log-space for numerical stability.
        log_joint = parts[0]
        for log_p in parts[1:]:
            log_joint = log_joint + log_p
        log_z = jax.nn.logsumexp(log_joint)
        fallback = jnp.ones_like(log_joint) / self.joint_vocab_size

        norm_j = jax.lax.cond(
            jnp.isfinite(log_z),
            lambda _: jnp.exp(log_joint - log_z),
            lambda _: fallback,
            operand=None,
        )

        return norm_j.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jax.Array, ...],
        context: ConditionalContext,
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
