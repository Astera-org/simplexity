"""JAX NumPy utilities."""

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.exceptions import DeviceResolutionError


def resolve_jax_device(backend: str | None = None) -> jax.Device:  # type: ignore[valid-type]
    """Resolve device specification to actual JAX device.

    Args:
        backend: Device specification string. Can be "auto", "cpu", "gpu", "cuda", "metal", "mps".

    Returns:
        JAX device.

    Raises:
        DeviceResolutionError: If the requested device is not available.
    """
    gpus = jax.devices("gpu")
    if backend in ("gpu", "cuda"):
        if not gpus:
            raise DeviceResolutionError("GPU requested but not available")
        return gpus[0]

    cpus = jax.devices("cpu")
    if backend == "cpu":
        if not cpus:
            raise DeviceResolutionError("CPU requested but not available")
        return cpus[0]

    if backend is None or backend == "auto":
        if gpus:
            return gpus[0]
        if cpus:
            return cpus[0]
        raise DeviceResolutionError("No devices available")

    raise DeviceResolutionError(f"Unknown device specification: {backend}")


@eqx.filter_jit
def entropy(probs: jax.Array, log: bool = False) -> jax.Array:
    """Compute the entropy of a log probability distribution."""
    if log:
        terms = -jnp.exp(probs) * probs
    else:
        terms = -probs * jnp.log(probs)
    terms = jnp.where(jnp.isnan(terms), jnp.zeros_like(terms), terms)
    return jnp.sum(terms)


@eqx.filter_jit
def log_matmul(log_A: jax.Array, log_B: jax.Array) -> jax.Array:
    """Compute the log of the matrix product of A and B.

    A and B are log-space matrices.
    """
    sum_mat = log_A[:, :, None] + log_B[None, :, :]
    return jax.nn.logsumexp(sum_mat, axis=1)


@eqx.filter_jit
def signed_logsumexp(
    log_abs_values: jax.Array, signs: jax.Array, axis: int | None = None
) -> tuple[jax.Array, jax.Array]:
    """Compute the log-sum-exp for a signed log-space array."""
    m = jnp.max(log_abs_values, axis=axis, keepdims=True)
    summation = jnp.sum(signs * jnp.exp(log_abs_values - m), axis=axis)
    m = jnp.squeeze(m, axis=axis)
    log_abs_values = m + jnp.log(jnp.abs(summation))
    signs = jnp.sign(summation)
    return log_abs_values, signs


class LogArray(eqx.Module):
    """A log-space array."""

    log_abs_values: jax.Array

    @classmethod
    def from_values(cls, values: jax.Array) -> "LogArray":
        """Create a LogArray from an array of values."""
        return cls(jnp.log(jnp.abs(values)))

    @eqx.filter_jit
    def logsumexp(self, axis: int | None = None) -> "LogArray":
        """Compute the log-sum-exp of the array."""
        return LogArray(jax.nn.logsumexp(self.log_abs_values, axis=axis))

    @eqx.filter_jit
    def __mul__(self, other: "LogArray") -> "LogArray":
        """Compute the product of two LogArrays."""
        return LogArray(self.log_abs_values + other.log_abs_values)

    @eqx.filter_jit
    def __matmul__(self, other: "LogArray") -> "LogArray":
        """Dispatch to appropriate multiplication based on input shapes."""
        if self.log_abs_values.ndim == 2 and other.log_abs_values.ndim == 2:
            return self.matmatmul(other)
        elif self.log_abs_values.ndim == 2 and other.log_abs_values.ndim == 1:
            return self.matvecmul(other)
        elif self.log_abs_values.ndim == 1 and other.log_abs_values.ndim == 2:
            return self.vecmatmul(other)
        else:
            raise ValueError("Unsupported shapes for matrix multiplication")

    @eqx.filter_jit
    def matmatmul(self, other: "LogArray") -> "LogArray":
        """Compute the product of two LogArrays."""
        log_abs_values = log_matmul(self.log_abs_values, other.log_abs_values)
        return LogArray(log_abs_values)

    @eqx.filter_jit
    def matvecmul(self, other: "LogArray") -> "LogArray":
        """Compute the product of a LogArray and a LogArray."""
        log_abs_values = jax.nn.logsumexp(self.log_abs_values + other.log_abs_values, axis=1)
        return LogArray(log_abs_values)

    @eqx.filter_jit
    def vecmatmul(self, other: "LogArray") -> "LogArray":
        """Compute the product of a LogArray and a LogArray."""
        log_abs_values = jax.nn.logsumexp(self.log_abs_values[:, None] + other.log_abs_values, axis=0)
        return LogArray(log_abs_values)


class SignedLogArray(eqx.Module):
    """A log-space array with a sign."""

    log_abs_values: jax.Array
    signs: jax.Array

    @classmethod
    def from_values(cls, values: jax.Array) -> "SignedLogArray":
        """Create a SignedLogArray from an array of values."""
        return cls(jnp.log(jnp.abs(values)), jnp.sign(values))

    @eqx.filter_jit
    def logsumexp(self, axis: int | None = None) -> "SignedLogArray":
        """Compute the log-sum-exp of the array."""
        return SignedLogArray(*signed_logsumexp(self.log_abs_values, self.signs, axis))

    @eqx.filter_jit
    def __mul__(self, other: "SignedLogArray") -> "SignedLogArray":
        """Compute the product of two SignedLogArrays."""
        return SignedLogArray(self.log_abs_values + other.log_abs_values, self.signs * other.signs)

    @eqx.filter_jit
    def __matmul__(self, other: "SignedLogArray") -> "SignedLogArray":
        """Dispatch to appropriate multiplication based on input shapes."""
        if self.log_abs_values.ndim == 2 and other.log_abs_values.ndim == 2:
            return self.matmatmul(other)
        elif self.log_abs_values.ndim == 2 and other.log_abs_values.ndim == 1:
            return self.matvecmul(other)
        elif self.log_abs_values.ndim == 1 and other.log_abs_values.ndim == 2:
            return self.vecmatmul(other)
        else:
            raise ValueError("Unsupported shapes for matrix multiplication")

    @eqx.filter_jit
    def matmatmul(self, other: "SignedLogArray") -> "SignedLogArray":
        """Compute the product of two SignedLogArrays."""
        log_abs_values = self.log_abs_values[:, :, None] + other.log_abs_values[None, :, :]
        signs = self.signs[:, :, None] * other.signs[None, :, :]
        return SignedLogArray(log_abs_values, signs).logsumexp(axis=1)

    @eqx.filter_jit
    def matvecmul(self, other: "SignedLogArray") -> "SignedLogArray":
        """Compute the product of a SignedLogArray and a SignedLogArray."""
        log_abs_values = self.log_abs_values + other.log_abs_values
        signs = self.signs * other.signs
        return SignedLogArray(log_abs_values, signs).logsumexp(axis=1)

    @eqx.filter_jit
    def vecmatmul(self, other: "SignedLogArray") -> "SignedLogArray":
        """Compute the product of a SignedLogArray and a SignedLogArray."""
        log_abs_values = self.log_abs_values[:, None] + other.log_abs_values
        signs = self.signs * other.signs
        return SignedLogArray(log_abs_values, signs).logsumexp(axis=1)
