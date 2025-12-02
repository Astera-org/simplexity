"""Sequential conditional structure: one-way conditional dependencies between factors.

Factor i>0 selects its parameter variant based on the emitted token
of factor i-1 (parent) via a control map.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.factored_beliefs.topology.topology import ConditionalContext
from simplexity.utils.factoring_utils import compute_obs_dist_for_variant


class SequentialConditional(eqx.Module):
    """Sequential conditional structure (autoregressive chain).

    Factors form a chain: Factor i depends on Factor i-1's emitted token.
    - Factor 0 always uses variant 0
    - Factor i>0 uses control_maps[i][parent_token] to select variant

    Joint distribution: P(t0, t1, ..., tF) = P(t0) * P(t1|t0) * ... * P(tF|t_{F-1})

    Attributes:
        control_maps: Tuple of F arrays. control_maps[i] has shape [V_{i-1}] for i>0,
            mapping parent token to variant index. control_maps[0] is None.
        vocab_sizes_py: Python int tuple of vocab sizes (for reshape operations)
    """

    control_maps: tuple[jnp.ndarray | None, ...]
    vocab_sizes_py: tuple[int, ...] | None

    def __init__(
        self,
        control_maps: tuple[jnp.ndarray | None, ...],
        vocab_sizes: jnp.ndarray | None = None,
    ):
        """Initialize sequential conditional structure.

        Args:
            control_maps: Control maps for variant selection. control_maps[0]
                should be None (root factor). control_maps[i] for i>0 should
                have shape [V_{i-1}] mapping parent token to variant index.
            vocab_sizes: Optional vocab sizes for shape operations. If provided,
                should be array of shape [F].
        """
        self.control_maps = tuple(control_maps)
        self.vocab_sizes_py = tuple(int(v) for v in vocab_sizes) if vocab_sizes is not None else None

    def compute_joint_distribution(self, context: ConditionalContext) -> jnp.ndarray:
        """Compute joint distribution using sequential factorization.

        Builds P(t0, t1, ..., tF) = P(t0) * P(t1|t0) * ... * P(tF|t_{F-1})
        iteratively, then flattens to radix encoding.

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

        # Root distribution (factor 0, variant 0)
        T_root = transition_matrices[0][0]  # [V_0, S_0, S_0]
        norm_root = normalizing_eigenvectors[0][0] if component_types[0] == "ghmm" else None
        p_root = compute_obs_dist_for_variant(
            component_types[0], states[0], T_root, norm_root
        )  # [V_0]
        joint = p_root

        # Iteratively extend with conditional factors
        for i in range(1, F):
            # Compute distributions for all variants of factor i
            K_i = num_variants[i]
            ks = jnp.arange(K_i, dtype=jnp.int32)

            # Vectorize over variants
            def get_dist_i(k: jnp.ndarray) -> jnp.ndarray:
                T_k = transition_matrices[i][k]
                norm_k = normalizing_eigenvectors[i][k] if component_types[i] == "ghmm" else None
                return compute_obs_dist_for_variant(component_types[i], states[i], T_k, norm_k)

            all_pi = jax.vmap(get_dist_i)(ks)  # [K_i, V_i]

            # Build conditional matrix [V_{i-1}, V_i] via control map
            cm = self.control_maps[i]  # [V_{i-1}]
            cond = all_pi[cm]  # [V_{i-1}, V_i]

            # Extend joint distribution
            # Current joint has shape [..., V_{i-1}]
            # We want to expand to [..., V_{i-1}, V_i]
            # Use precomputed Python ints for reshape (JIT-compatible)
            if self.vocab_sizes_py is not None:
                prev_vocab_size = self.vocab_sizes_py[i - 1]
                curr_vocab_size = self.vocab_sizes_py[i]
            else:
                prev_vocab_size = int(context.vocab_sizes[i - 1])
                curr_vocab_size = int(context.vocab_sizes[i])
            left = joint.reshape(-1, prev_vocab_size)  # [P, V_{i-1}]
            extended = left[..., None] * cond[None, ...]  # [P, V_{i-1}, V_i]
            joint = extended.reshape(joint.shape + (curr_vocab_size,))

        return joint.reshape(-1)

    def select_variants(
        self,
        obs_tuple: tuple[jnp.ndarray, ...],
        context: ConditionalContext,
    ) -> tuple[jnp.ndarray, ...]:
        """Select variants based on parent tokens in chain.

        Args:
            obs_tuple: Tuple of observed tokens (one per factor)
            context: Conditional context (unused for sequential conditional)

        Returns:
            Tuple of variant indices (one per factor)
        """
        variants = []
        for i in range(len(obs_tuple)):
            if i == 0:
                # Root factor always uses variant 0
                variants.append(jnp.array(0, dtype=jnp.int32))
            else:
                # Select based on parent's observed token
                parent_token = obs_tuple[i - 1]
                k_i = self.control_maps[i][parent_token] # type: ignore
                variants.append(k_i)
        return tuple(variants)

    def get_required_params(self) -> dict[str, type]:
        """Return required parameters for sequential conditional structure."""
        return {"control_maps": tuple}
