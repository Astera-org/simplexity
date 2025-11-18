from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def make_prefix_groups(inputs: jax.Array) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Group positions by prefix of tokens.

    Args:
        inputs: (batch, seq_len) integer token ids

    Returns:
        dict: prefix_tuple -> list of (seq_idx, pos) positions
    """
    batch_size, seq_len = inputs.shape
    prefix_to_indices: dict[tuple[int, ...], list[tuple[int, int]]] = {}

    inputs_np = np.asarray(inputs)

    for seq_idx in range(batch_size):
        seq = inputs_np[seq_idx]
        for pos in range(seq_len):
            prefix = tuple(seq[:pos + 1])
            prefix_to_indices.setdefault(prefix, []).append((seq_idx, pos))

    return prefix_to_indices


def dedup_tensor_first(
    tensor: jax.Array,
    prefix_to_indices: dict[tuple[int, ...], list[tuple[int, int]]],
) -> tuple[jax.Array, list[tuple[int, ...]]]:
    """Deduplicate a (batch, seq_len, ...) tensor by prefixes, taking the first occurrence.

    Returns:
        dedup_values: (num_prefixes, ...) tensor
        prefixes: list of prefix tuples in the same order
    """
    values = []
    prefixes: list[tuple[int, ...]] = []

    for prefix, idxs in prefix_to_indices.items():
        seq_idx, pos = idxs[0]
        values.append(tensor[seq_idx, pos])
        prefixes.append(prefix)

    return jnp.stack(values, axis=0), prefixes


def dedup_probs_sum(
    probs: jax.Array,
    prefix_to_indices: dict[tuple[int, ...], list[tuple[int, int]]],
) -> tuple[jax.Array, list[tuple[int, ...]]]:
    """Deduplicate (batch, seq_len) probabilities by summing over all occurrences of each prefix."""
    dedup_values = []
    prefixes: list[tuple[int, ...]] = []

    probs_np = np.asarray(probs)

    for prefix, idxs in prefix_to_indices.items():
        total = 0.0
        for seq_idx, pos in idxs:
            total += float(probs_np[seq_idx, pos])
        dedup_values.append(total)
        prefixes.append(prefix)

    dedup_probs = jnp.array(dedup_values, dtype=probs.dtype)
    # normalize to sum to 1
    total_mass = dedup_probs.sum()
    if total_mass > 0:
        dedup_probs = dedup_probs / total_mass

    return dedup_probs, prefixes


@dataclass
class PrefixDataset:
    """A clean container for prefix-level data.

    All tensors are shape (N, ...), where N = #unique prefixes.
    """

    prefixes: list[tuple[int, ...]]
    beliefs: jax.Array  # (N, B)
    probs: jax.Array  # (N,)
    activations_by_layer: dict[str, jax.Array]  # layer_name -> (N, d)


def build_prefix_dataset(
    inputs: jax.Array,  # (batch, seq_len)
    beliefs: jax.Array,  # (batch, seq_len, B)
    probs: jax.Array,  # (batch, seq_len)
    activations_by_layer: dict[str, jax.Array],  # layer -> (batch, seq_len, d_layer)
) -> PrefixDataset:
    """Deduplicate everything by prefix.

    - group positions with the same prefix
    - sum probs per prefix
    - take first beliefs & activations per prefix
    """
    prefix_to_indices = make_prefix_groups(inputs)

    # Dedup beliefs & probs
    dedup_beliefs, prefixes = dedup_tensor_first(beliefs, prefix_to_indices)
    dedup_probs, prefixes2 = dedup_probs_sum(probs, prefix_to_indices)

    # Sanity check: order should match
    assert prefixes == prefixes2, "Internal prefix ordering mismatch"

    # Dedup activations per layer
    dedup_acts_by_layer = {}
    for name, acts in activations_by_layer.items():
        dedup_acts, prefixes3 = dedup_tensor_first(acts, prefix_to_indices)
        assert prefixes3 == prefixes, f"Prefix mismatch for layer {name}"
        dedup_acts_by_layer[name] = dedup_acts

    return PrefixDataset(
        prefixes=prefixes,
        beliefs=dedup_beliefs,
        probs=dedup_probs,
        activations_by_layer=dedup_acts_by_layer,
    )