"""Noisy channel utilities for generative processes.

Implements observation-level noise by blurring transition matrices,
replacing each output symbol with probability epsilon to another
uniformly chosen symbol.
"""

import math

import jax
import jax.numpy as jnp


def apply_noisy_channel(transition_matrices: jax.Array, noise_epsilon: float) -> jax.Array:
    """Apply noisy channel blur to transition matrices.

    Replaces each output symbol with probability noise_epsilon to another
    uniformly chosen symbol.

    Args:
        transition_matrices: Transition matrices of shape [V, S, S] where V is vocab_size.
        noise_epsilon: Noise probability in [0, 1]. 0 means no noise, 1 means uniform noise.

    Returns:
        Blurred transition matrices of shape [V, S, S].

    Raises:
        ValueError: If noise_epsilon is not in [0, 1].
    """
    if not 0.0 <= noise_epsilon <= 1.0:
        raise ValueError(f"noise_epsilon must be in [0, 1], got {noise_epsilon}")

    if noise_epsilon == 0.0:
        return transition_matrices

    vocab_size = transition_matrices.shape[0]
    blur_matrix = (1.0 - noise_epsilon) * jnp.eye(vocab_size) + noise_epsilon * (
        jnp.ones((vocab_size, vocab_size)) / vocab_size
    )
    return jnp.einsum("kij, kn -> nij", transition_matrices, blur_matrix)


def compute_joint_blur_matrix(vocab_sizes: tuple[int, ...], noise_epsilon: float) -> jax.Array:
    """Compute blur matrix for joint observation space.

    For factored processes with joint noise, creates a blur matrix
    over the composite vocabulary.

    Args:
        vocab_sizes: Tuple of vocab sizes per factor.
        noise_epsilon: Noise probability in [0, 1].

    Returns:
        Blur matrix of shape [joint_vocab, joint_vocab] where joint_vocab = prod(vocab_sizes).

    Raises:
        ValueError: If noise_epsilon is not in [0, 1].
    """
    if not 0.0 <= noise_epsilon <= 1.0:
        raise ValueError(f"noise_epsilon must be in [0, 1], got {noise_epsilon}")

    joint_vocab = math.prod(vocab_sizes)

    if noise_epsilon == 0.0:
        return jnp.eye(joint_vocab)

    return (1.0 - noise_epsilon) * jnp.eye(joint_vocab) + noise_epsilon * (
        jnp.ones((joint_vocab, joint_vocab)) / joint_vocab
    )
