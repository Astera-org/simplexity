"""Tree conditional structure: arbitrary parent dependencies between factors.

Generalizes SequentialConditional to allow factor i to depend on any factor j < i
(not just i-1), enabling fan-out structures like:

    Factor 0 (root)
       /      \
      v        v
   Factor 1  Factor 2

Where both children depend on the same parent but not on each other.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.structures.protocol import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class TreeConditional(eqx.Module):
    """Tree-structured conditional dependencies.

    Generalizes SequentialConditional to allow arbitrary parent dependencies.
    Each factor i can depend on any factor j < i (not just i-1).

    Joint distribution: P(t0, t1, ..., tF) = P(t0) * P(t1|t_{p1}) * ... * P(tF|t_{pF})
    where p_i is the parent of factor i.

    Attributes:
        parent_indices: Tuple of F ints/Nones. parent_indices[i] = parent factor index
            for factor i, or None if factor i is a root.
        control_maps: Tuple of F arrays. control_maps[i] has shape [V_{parent_i}] for
            non-root factors, mapping parent token to variant index. control_maps[i]
            is None for root factors.
        vocab_sizes_py: Python int tuple of vocab sizes (for reshape operations)
    """

    parent_indices: tuple[int | None, ...]
    control_maps: tuple[jax.Array | None, ...]
    vocab_sizes_py: tuple[int, ...]

    def __init__(
        self,
        parent_indices: tuple[int | None, ...],
        control_maps: tuple[jax.Array | None, ...],
        vocab_sizes: jax.Array,
    ):
        """Initialize tree conditional structure.

        Args:
            parent_indices: Parent indices for each factor. parent_indices[i] should be
                None for root factors, or an integer j < i for non-root factors.
            control_maps: Control maps for variant selection. control_maps[i] should be
                None for root factors. control_maps[i] for non-root factors should have
                shape [V_{parent_i}] mapping parent token to variant index.
            vocab_sizes: Vocab sizes for shape operations. Must be array of shape [F].
        """
        self.parent_indices = tuple(parent_indices)
        self.control_maps = tuple(control_maps)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes)

    def compute_joint_distribution(self, context: ConditionalContext) -> jax.Array:
        """Compute joint distribution using tree factorization.

        Builds P(t0, t1, ..., tF) iteratively using parent dependencies,
        then flattens to radix encoding.

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
        num_variants = context.num_variants

        # Root distribution (factor 0, variant 0)
        transition_matrix_root = transition_matrices[0][0]  # [V_0, S_0, S_0]
        norm_root = normalizing_eigenvectors[0][0] if component_types[0] == "ghmm" else None
        p_root = compute_obs_dist_for_variant(component_types[0], states[0], transition_matrix_root, norm_root)  # [V_0]
        joint = p_root

        # Iteratively extend with conditional factors
        for i in range(1, num_factors):
            parent_idx = self.parent_indices[i]

            # Compute distributions for all variants of factor i
            num_var_i = num_variants[i]
            ks = jnp.arange(num_var_i, dtype=jnp.int32)

            # Vectorize over variants
            def get_dist_i(k: jax.Array, i: int = i) -> jax.Array:
                transition_matrix_k = transition_matrices[i][k]
                norm_k = normalizing_eigenvectors[i][k] if component_types[i] == "ghmm" else None
                return compute_obs_dist_for_variant(component_types[i], states[i], transition_matrix_k, norm_k)

            all_pi = jax.vmap(get_dist_i)(ks)  # [K_i, V_i]

            # Build conditional matrix [V_{parent}, V_i] via control map
            cm = self.control_maps[i]  # [V_{parent}]
            cond = all_pi[cm]  # [V_{parent}, V_i]

            # Extend joint distribution
            # Current joint has shape [V_0, V_1, ..., V_{i-1}]
            # We want to multiply by P(t_i | t_{parent})
            # The parent is at axis position parent_idx

            # Reshape cond for broadcasting:
            # We need cond to have shape [1]*parent_idx + [V_parent] + [1]*(i-1-parent_idx) + [V_i]
            parent_vocab_size = self.vocab_sizes_py[parent_idx]  # type: ignore[index]
            curr_vocab_size = self.vocab_sizes_py[i]

            # Build broadcast shape: [1, ..., 1, V_parent, 1, ..., 1, V_i]
            broadcast_shape = [1] * parent_idx + [parent_vocab_size] + [1] * (i - 1 - parent_idx) + [curr_vocab_size]  # type: ignore[operator]
            cond_broadcast = cond.reshape(broadcast_shape)

            # Extend joint: [..., V_{i-1}] -> [..., V_{i-1}, V_i]
            joint = joint[..., None] * cond_broadcast

        return joint.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jax.Array, ...],
        context: ConditionalContext,  # pylint: disable=unused-argument
    ) -> tuple[jax.Array, ...]:
        """Select variants based on parent tokens in tree.

        Args:
            obs_tuple: Tuple of observed tokens (one per factor)
            context: Conditional context (unused for tree conditional)

        Returns:
            Tuple of variant indices (one per factor)
        """
        variants = []
        for i in range(len(obs_tuple)):
            parent_idx = self.parent_indices[i]
            if parent_idx is None:
                # Root factor always uses variant 0
                variants.append(jnp.array(0, dtype=jnp.int32))
            else:
                # Select based on parent's observed token
                parent_token = obs_tuple[parent_idx]
                k_i = self.control_maps[i][parent_token]  # type: ignore
                variants.append(k_i)
        return tuple(variants)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters for tree conditional structure."""
        return {"parent_indices": tuple, "control_maps": tuple}
