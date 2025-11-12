"""Coupling topology protocol for factored generative processes.

Defines the interface for different coupling strategies between factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import jax.numpy as jnp

ComponentType = Literal["hmm", "ghmm"]
FactoredState = tuple[jnp.ndarray, ...]


@dataclass
class CouplingContext:
    """Context information needed by topologies to compute joint distributions.

    Attributes:
        states: Tuple of state vectors, one per factor (shape [S_i])
        component_types: Type of each factor ("hmm" or "ghmm")
        transition_matrices: Per-factor transition tensors (shape [K_i, V_i, S_i, S_i])
        normalizing_eigenvectors: Per-factor eigenvectors (shape [K_i, S_i])
        vocab_sizes: Vocabulary size per factor (shape [F])
        num_variants: Number of parameter variants per factor
    """

    states: FactoredState
    component_types: tuple[ComponentType, ...]
    transition_matrices: tuple[jnp.ndarray, ...]
    normalizing_eigenvectors: tuple[jnp.ndarray, ...]
    vocab_sizes: jnp.ndarray
    num_variants: tuple[int, ...]


class CouplingTopology(Protocol):
    """Protocol for coupling topologies between factors.

    A coupling topology defines how factors interact to produce joint
    observation distributions and how variant selection works.
    """

    def compute_joint_distribution(self, context: CouplingContext) -> jnp.ndarray:
        """Compute joint observation distribution across all factors.

        Args:
            context: All information needed to compute the joint distribution

        Returns:
            Flattened joint distribution of shape [prod(V_i)]
            Uses radix encoding: token = sum(t_i * prod(V_j for j>i))
        """
        ...

    def select_variants(
        self,
        obs_tuple: tuple[jnp.ndarray, ...],
        context: CouplingContext,
    ) -> tuple[jnp.ndarray, ...]:
        """Select parameter variant for each factor given observations.

        Args:
            obs_tuple: Tuple of F observed tokens (one per factor)
            context: Context information

        Returns:
            Tuple of F variant indices (one per factor)
        """
        ...

    def get_required_params(self) -> dict[str, type]:
        """Return dictionary of topology-specific parameters and their types.

        Returns:
            Dict mapping parameter names to expected types
            (e.g., {"control_maps": tuple} for chain topology)
        """
        ...
