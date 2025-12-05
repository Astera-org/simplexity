"""Conditional structure protocol for factored generative processes.

Defines the interface for different conditional dependency structures between factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import jax

ComponentType = Literal["hmm", "ghmm"]
FactoredState = tuple[jax.Array, ...]


@dataclass
class ConditionalContext:
    """Context information needed by conditional structures to compute joint distributions.

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
    transition_matrices: tuple[jax.Array, ...]
    normalizing_eigenvectors: tuple[jax.Array, ...]
    vocab_sizes: jax.Array
    num_variants: tuple[int, ...]


class ConditionalStructure(Protocol):
    """Protocol for conditional dependency structures between factors.

    A conditional structure defines how factors conditionally depend on each other
    to produce joint observation distributions and how variant selection works.
    """

    def compute_joint_distribution(self, context: ConditionalContext) -> jax.Array:
        """Compute the joint distribution of the conditional structure."""
        ...  # pylint: disable=unnecessary-ellipsis

    def select_variants(
        self,
        obs_tuple: tuple[jax.Array, ...],
        context: ConditionalContext,
    ) -> tuple[jax.Array, ...]:
        """Select the variants of the conditional structure."""
        ...  # pylint: disable=unnecessary-ellipsis

    def get_required_params(self) -> dict[str, type]:
        """Get the required parameters for the conditional structure."""
        ...  # pylint: disable=unnecessary-ellipsis
