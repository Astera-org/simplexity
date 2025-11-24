import jax
import jax.numpy as jnp
import numpy as np


def _standardize_features(x: np.ndarray | jax.Array) -> jax.Array:
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.ndim != 2:
        raise ValueError("Features must be a 2D array")
    return x_arr


def _standardize_targets(y: np.ndarray | jax.Array) -> jax.Array:
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    if y_arr.ndim != 2:
        raise ValueError("Targets must be a 2D array")
    return y_arr


def _normalize_weights(weights: np.ndarray | jax.Array | None, n_samples: int) -> jax.Array | None:
    """Normalize weights to sum to 1, or return None if weights is None."""
    if weights is None:
        return None
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if weights.ndim != 1 or weights.shape[0] != n_samples:
        raise ValueError("Weights must be shape (n_samples,)")
    if bool(jnp.any(weights < 0)):
        raise ValueError("Weights must be non-negative")
    total = float(weights.sum())
    if not jnp.isfinite(total) or total <= 0:
        raise ValueError("Sum of weights must be positive")
    return jnp.asarray(weights / total)
