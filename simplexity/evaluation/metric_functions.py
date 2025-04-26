from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class MetricFunction(Protocol):
    """A function that computes a eval metric from a model's predicted logits and the ground truth labels."""

    def __call__(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Compute the eval metric."""
        ...


@eqx.filter_jit
def loss_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute loss."""
    return jnp.array(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


@eqx.filter_jit
def accuracy_fn(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute accuracy."""
    preds = jnp.argmax(logits)
    return preds == labels
