import jax
import jax.numpy as jnp


def principal_ev(T: jax.Array) -> jax.Array:
    eigenvalues, eigenvectors = jnp.linalg.eig(T)
    i = jnp.argmax(jnp.abs(eigenvalues))
    vector = jnp.real(eigenvectors[:, i])
    sign = jnp.where(jnp.sum(vector) < 0, -1.0, 1.0)
    vector = vector * sign
    return vector / jnp.mean(vector)


def pad(arr: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
    pad_width = [(0, target - current) for target, current in zip(target_shape, arr.shape, strict=True)]
    return jnp.pad(arr, pad_width, mode="constant", constant_values=0)


def stack(arrs: list[jax.Array]) -> jax.Array:
    shapes = [arr.shape for arr in arrs]
    max_shape = tuple(max(shape_i) for shape_i in zip(*shapes, strict=True))
    return jnp.stack([pad(arr, max_shape) for arr in arrs])


def mixed_radix_weights(Vs: jax.Array) -> jax.Array:
    return jnp.roll(jnp.cumprod(Vs), 1).at[0].set(1)


def mixed_radix_encode(x_factors: jax.Array, *, weights: jax.Array) -> jax.Array:
    return jnp.sum(x_factors * weights)


def mixed_radix_decode(x: jax.Array, *, Vs: jax.Array, weights: jax.Array) -> jax.Array:
    return (x // weights) % Vs
