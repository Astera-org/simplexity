from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def make_prefix_groups(inputs: jax.Array) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Group positions by prefix of tokens."""
    batch_size, seq_len = inputs.shape
    prefix_to_indices = defaultdict(list)

    inputs_np = np.asarray(inputs)

    for seq_idx in range(batch_size):
        seq = inputs_np[seq_idx]
        for pos in range(seq_len):
            prefix = tuple(seq[: pos + 1])
            prefix_to_indices[prefix].append((seq_idx, pos))

    return prefix_to_indices


def dedup_tensor_first(
    tensor: jax.Array,
    prefix_to_indices: dict[tuple[int, ...], list[tuple[int, int]]],
) -> tuple[jax.Array, list[tuple[int, ...]]]:
    """Deduplicate a (batch, seq_len, ...) tensor by prefixes, taking the first occurrence."""
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


def make_sequence_groups(inputs: jax.Array) -> dict[tuple[int, ...], list[int]]:
    """Group sequences by full sequence.

    Args:
        inputs: (batch, seq_len) integer token ids

    Returns:
        dict: sequence_tuple -> list of seq_idx indices with that sequence
    """
    batch_size, _ = inputs.shape
    sequence_to_indices: dict[tuple[int, ...], list[int]] = {}

    inputs_np = np.asarray(inputs)

    for seq_idx in range(batch_size):
        seq = tuple(inputs_np[seq_idx])
        sequence_to_indices.setdefault(seq, []).append(seq_idx)

    return sequence_to_indices


def dedup_last_token_tensor_first(
    tensor: jax.Array,
    sequence_to_indices: dict[tuple[int, ...], list[int]],
) -> tuple[jax.Array, list[tuple[int, ...]]]:
    """Deduplicate a (batch, ...) tensor by full sequences, taking the first occurrence."""
    values = []
    sequences: list[tuple[int, ...]] = []

    for seq, idxs in sequence_to_indices.items():
        seq_idx = idxs[0]
        values.append(tensor[seq_idx])
        sequences.append(seq)

    return jnp.stack(values, axis=0), sequences


def dedup_last_token_probs_sum(
    probs: jax.Array,
    sequence_to_indices: dict[tuple[int, ...], list[int]],
) -> tuple[jax.Array, list[tuple[int, ...]]]:
    """Deduplicate (batch,) probabilities by summing over all occurrences of each sequence."""
    dedup_values = []
    sequences: list[tuple[int, ...]] = []

    probs_np = np.asarray(probs)

    for seq, idxs in sequence_to_indices.items():
        total = sum(float(probs_np[idx]) for idx in idxs)
        dedup_values.append(total)
        sequences.append(seq)

    dedup_probs = jnp.array(dedup_values, dtype=probs.dtype)
    # normalize to sum to 1
    total_mass = dedup_probs.sum()
    if total_mass > 0:
        dedup_probs = dedup_probs / total_mass

    return dedup_probs, sequences


@dataclass
class DeduplicatedDataset:
    """A clean container for last-token-only data."""

    sequences: list[tuple[int, ...]]
    beliefs: jax.Array
    probs: jax.Array
    activations_by_layer: dict[str, jax.Array]


def build_deduplicated_dataset(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
    select_last_token: bool = False,
) -> DeduplicatedDataset:
    """Deduplicate everything by prefix."""
    if select_last_token:
        return build_last_token_dataset(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
        )
    else:
        return build_prefix_dataset(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
        )


def build_prefix_dataset(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
) -> DeduplicatedDataset:
    """Deduplicate everything by prefix."""
    prefix_to_indices = make_prefix_groups(inputs)

    dedup_beliefs, prefixes = dedup_tensor_first(beliefs, prefix_to_indices)
    dedup_probs, prefixes2 = dedup_probs_sum(probs, prefix_to_indices)

    if prefixes != prefixes2:
        raise ValueError("Internal prefix ordering mismatch")

    dedup_acts_by_layer = {}
    for name, acts in activations_by_layer.items():
        dedup_acts, prefixes3 = dedup_tensor_first(acts, prefix_to_indices)
        if prefixes3 != prefixes:
            raise ValueError(f"Prefix mismatch for layer {name}")
        dedup_acts_by_layer[name] = dedup_acts

    return DeduplicatedDataset(
        sequences=prefixes,
        beliefs=dedup_beliefs,
        probs=dedup_probs,
        activations_by_layer=dedup_acts_by_layer,
    )


def build_last_token_dataset(
    inputs: jax.Array,
    beliefs: jax.Array,
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
) -> DeduplicatedDataset:
    """Deduplicate everything by full sequence."""
    sequence_to_indices = make_sequence_groups(inputs)

    # Dedup beliefs & probs
    dedup_beliefs, sequences = dedup_last_token_tensor_first(beliefs, sequence_to_indices)
    dedup_probs, sequences2 = dedup_last_token_probs_sum(probs, sequence_to_indices)

    if sequences != sequences2:
        raise ValueError("Internal sequence ordering mismatch")

    # Dedup activations per layer
    dedup_acts_by_layer = {}
    for name, acts in activations_by_layer.items():
        dedup_acts, sequences3 = dedup_last_token_tensor_first(acts, sequence_to_indices)
        if sequences3 != sequences:
            raise ValueError(f"Sequence mismatch for layer {name}")
        dedup_acts_by_layer[name] = dedup_acts

    return DeduplicatedDataset(
        sequences=sequences,
        beliefs=dedup_beliefs,
        probs=dedup_probs,
        activations_by_layer=dedup_acts_by_layer,
    )
