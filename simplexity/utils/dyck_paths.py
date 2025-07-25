import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def catalan_number(n: int) -> int:
    """Calculates the nth Catalan number."""
    return math.comb(2 * n, n) // (n + 1)


# A JIT-compatible helper function to calculate log(n choose k)
# Using gammaln is more numerically stable than direct factorial calculation.
@eqx.filter_jit
def log_comb(n: jax.Array, k: jax.Array) -> jax.Array:
    """Calculates log(n choose k)."""
    # n choose k is 0 if k < 0 or k > n.
    is_out_of_bounds = (k < 0) | (k > n)

    # Use lax.cond to avoid evaluating gammaln on invalid inputs, which would be NaN.
    # The lambda functions ensure that the code is only executed when the condition is met.
    return jax.lax.cond(
        is_out_of_bounds,
        lambda: -jnp.inf,  # log(0) is -inf
        lambda: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1),
    )


@eqx.filter_jit
def num_paths_from(up_steps: jax.Array, down_steps: jax.Array, n: int) -> jax.Array:
    """Calculates the number of valid Dyck paths from a given state.

    This function computes the number of paths from the current position
    (up_steps, down_steps) to the end (n, n) that do not go below the diagonal.
    This is calculated using the reflection principle.

    Args:
        up_steps: The number of up-steps taken so far.
        down_steps: The number of down-steps taken so far.
        n: The half-length of the total path.

    Returns:
        The total number of valid continuations from the current state.
    """
    # If we can't possibly complete the path, return 0.
    # This handles cases where we are already off the grid.
    is_invalid = (up_steps > n) | (down_steps > up_steps)

    remaining_steps = 2 * n - up_steps - down_steps
    remaining_up = n - up_steps

    # Number of paths from (up, down) to (n, n)
    total_paths = jnp.exp(log_comb(remaining_steps, remaining_up))

    # Number of invalid paths are those that touch y=-1. By the reflection principle,
    # this is the number of paths to the reflection of the endpoint across y=-1.
    # The number of up steps needed for an invalid path is remaining_up - 1.
    invalid_paths = jnp.exp(log_comb(remaining_steps, remaining_up - 1))  # <-- FIX: Was remaining_up + 1

    # The result is an integer, so we round it.
    # We use jnp.where to safely handle the invalid state.
    return jnp.where(is_invalid, 0, jnp.round(total_paths - invalid_paths)).astype(jnp.int32)


@eqx.filter_jit
def unrank_dyck_path(rank: jax.Array, n: int) -> jax.Array:
    """Generates the k-th Dyck path of length 2n using an unranking algorithm.

    Args:
        rank: The lexicographical rank of the path to generate (0-indexed).
        n: The half-length of the path.

    Returns:
        A JAX array representing the k-th Dyck path.
    """

    # The body function for our jax.lax.scan loop.
    def step(carry, _):
        path, up_steps, down_steps, current_rank = carry

        # Calculate how many valid paths start with an UP step from here.
        # This determines the size of the block of paths we are in.
        count = num_paths_from(up_steps + 1, down_steps, n)

        # Decide the next step based on the rank.
        # If current_rank < count, we must take an UP step to stay in this block.
        # Otherwise, we take a DOWN step and subtract `count` from the rank to find
        # our rank within the next block of paths.
        is_up_step = current_rank < count

        next_step = jnp.where(is_up_step, 1, -1)
        next_up = jnp.where(is_up_step, up_steps + 1, up_steps)
        next_down = jnp.where(is_up_step, down_steps, down_steps + 1)
        next_rank = jnp.where(is_up_step, current_rank, current_rank - count)

        # Update the path array at the current position.
        current_pos = up_steps + down_steps
        path = path.at[current_pos].set(next_step)

        return (path, next_up, next_down, next_rank), None

    # Initialize the state for the scan.
    initial_path = jnp.zeros(2 * n, dtype=jnp.int8)
    initial_carry = (initial_path, 0, 0, rank)  # path, up, down, rank

    # Run the scan over 2n steps to build the path.
    (final_path, _, _, _), _ = jax.lax.scan(step, initial_carry, None, length=2 * n)

    return final_path


@eqx.filter_jit
def get_dyck_paths(n: int) -> jax.Array:
    """Generates all Dyck paths of length 2n using a JIT-compiled function.

    Args:
        n: The half-length of the Dyck paths (number of up-steps).

    Returns:
        A JAX array containing all Dyck paths of length 2n. Each row
        represents a unique Dyck path.
    """
    if n == 0:
        return jnp.array([[]], dtype=jnp.int8)

    # The number of Dyck paths is the nth Catalan number.
    catalan_n = catalan_number(n)

    # Create an array of all ranks [0, 1, ..., C_n - 1].
    ranks = jnp.arange(catalan_n)

    # Use vmap to apply the JIT-compiled unranking function to all ranks in parallel.
    # We specify in_axes=(0, None) because we want to map over the first argument (rank)
    # but keep the second argument (n) constant for all calls.
    all_paths = eqx.filter_vmap(unrank_dyck_path, in_axes=(0, None))(ranks, n)

    return all_paths
