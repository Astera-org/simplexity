import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def compute_inputs_hash(inputs: jax.Array) -> str:
    """Compute a deterministic hash of the inputs array for cache keying.

    Args:
        inputs: (batch, seq_len) integer token ids

    Returns:
        A hex string hash that uniquely identifies the inputs content
    """
    inputs_np = np.asarray(inputs)
    return hashlib.sha256(inputs_np.tobytes()).hexdigest()


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


def dedup_tuple_of_tensors_first(
    tensors: tuple[jax.Array, ...],
    prefix_to_indices: dict[tuple[int, ...], list[tuple[int, int]]],
) -> tuple[tuple[jax.Array, ...], list[tuple[int, ...]]]:
    """Deduplicate a tuple of (batch, seq_len, ...) tensors by prefixes, taking the first occurrence in each tuple."""
    combined_values = []
    prefixes = prefix_to_indices.keys()

    for tensor in tensors:
        values = []
        for idxs in prefix_to_indices.values():
            seq_idx, pos = idxs[0]
            values.append(tensor[seq_idx, pos])
        combined_values.append(jnp.stack(values, axis=0))

    return tuple(combined_values), list(prefixes)


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

    total_mass = dedup_probs.sum()
    if total_mass > 0:
        dedup_probs = dedup_probs / total_mass
    else:
        raise ValueError("Total probability mass is zero after deduplication")

    return dedup_probs, prefixes


def make_sequence_groups(inputs: jax.Array) -> dict[tuple[int, ...], list[int]]:
    """Group sequences by full sequence.

    Args:
        inputs: (batch, seq_len) integer token ids

    Returns:
        dict: sequence_tuple -> list of seq_idx indices with that sequence
    """
    batch_size, _ = inputs.shape
    sequence_to_indices: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)

    inputs_np = np.asarray(inputs)

    for seq_idx in range(batch_size):
        seq = tuple(inputs_np[seq_idx])
        sequence_to_indices[seq].append(seq_idx)

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


def dedup_last_token_tuple_of_tensors_first(
    tensors: tuple[jax.Array, ...],
    sequence_to_indices: dict[tuple[int, ...], list[int]],
) -> tuple[tuple[jax.Array, ...], list[tuple[int, ...]]]:
    """Deduplicate a tuple of (batch, ...) tensors by full sequences, taking the first occurrence in each tuple."""
    combined_values = []
    sequences = list(sequence_to_indices.keys())

    for tensor in tensors:
        values = []
        for idxs in sequence_to_indices.values():
            seq_idx = idxs[0]
            values.append(tensor[seq_idx])
        combined_values.append(jnp.stack(values, axis=0))

    return tuple(combined_values), sequences


@dataclass
class DeduplicatedDataset:
    """A clean container for last-token-only data."""

    sequences: list[tuple[int, ...]]
    beliefs: jax.Array | tuple[jax.Array, ...]
    probs: jax.Array
    activations_by_layer: dict[str, jax.Array]


def build_deduplicated_dataset(
    inputs: jax.Array,
    beliefs: jax.Array | tuple[jax.Array, ...],
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
    select_last_token: bool = False,
    skip_first_token: bool = False,
) -> DeduplicatedDataset:
    """Deduplicate everything by prefix."""
    if select_last_token:
        return build_last_token_dataset(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
            skip_first_token=skip_first_token,
        )
    else:
        return build_prefix_dataset(
            inputs,
            beliefs,
            probs,
            activations_by_layer,
            skip_first_token=skip_first_token,
        )


def build_prefix_dataset(
    inputs: jax.Array,
    beliefs: jax.Array | tuple[jax.Array, ...],
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
    skip_first_token: bool = False,
) -> DeduplicatedDataset:
    """Deduplicate everything by prefix."""
    if skip_first_token:
        inputs = inputs[:, 1:]
        if isinstance(beliefs, tuple):
            beliefs = tuple(b[:, 1:, ...] for b in beliefs)
        else:
            beliefs = beliefs[:, 1:, ...]
        probs = probs[:, 1:]
        activations_by_layer = {name: acts[:, 1:, ...] for name, acts in activations_by_layer.items()}
    prefix_to_indices = make_prefix_groups(inputs)

    dedup_beliefs, prefixes = (
        dedup_tensor_first(beliefs, prefix_to_indices)
        if isinstance(beliefs, jax.Array)
        else dedup_tuple_of_tensors_first(beliefs, prefix_to_indices)
    )
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
    beliefs: jax.Array | tuple[jax.Array, ...],
    probs: jax.Array,
    activations_by_layer: dict[str, jax.Array],
    skip_first_token: bool = False,
) -> DeduplicatedDataset:
    """Deduplicate everything by full sequence."""
    if skip_first_token:
        inputs = inputs[:, 1:]
        if isinstance(beliefs, tuple):
            beliefs = tuple(b[:, 1:, ...] for b in beliefs)
        else:
            beliefs = beliefs[:, 1:, ...]
        probs = probs[:, 1:]
        activations_by_layer = {name: acts[:, 1:, ...] for name, acts in activations_by_layer.items()}
    if isinstance(beliefs, tuple):
        beliefs = tuple(b[:, -1, :] for b in beliefs)
    else:
        beliefs = beliefs[:, -1, :]
    probs = probs[:, -1]
    activations_by_layer = {name: acts[:, -1, :] for name, acts in activations_by_layer.items()}
    sequence_to_indices = make_sequence_groups(inputs)

    # Dedup beliefs & probs
    dedup_beliefs, sequences = (
        dedup_last_token_tensor_first(beliefs, sequence_to_indices)
        if isinstance(beliefs, jax.Array)
        else dedup_last_token_tuple_of_tensors_first(beliefs, sequence_to_indices)
    )
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


@dataclass
class SerializableDeduplicatedDataset:
    """A serializable version of DeduplicatedDataset using numpy arrays."""

    sequences: list[tuple[int, ...]]
    beliefs: np.ndarray | tuple[np.ndarray, ...]
    probs: np.ndarray
    activations_by_layer: dict[str, np.ndarray]

    @classmethod
    def from_deduplicated_dataset(cls, dataset: DeduplicatedDataset) -> "SerializableDeduplicatedDataset":
        """Convert a DeduplicatedDataset to a serializable format."""
        beliefs = (
            tuple(np.asarray(b) for b in dataset.beliefs)
            if isinstance(dataset.beliefs, tuple)
            else np.asarray(dataset.beliefs)
        )
        return cls(
            sequences=dataset.sequences,
            beliefs=beliefs,
            probs=np.asarray(dataset.probs),
            activations_by_layer={k: np.asarray(v) for k, v in dataset.activations_by_layer.items()},
        )

    def to_deduplicated_dataset(self) -> DeduplicatedDataset:
        """Convert back to a DeduplicatedDataset with JAX arrays."""
        beliefs = (
            tuple(jnp.asarray(b) for b in self.beliefs)
            if isinstance(self.beliefs, tuple)
            else jnp.asarray(self.beliefs)
        )
        return DeduplicatedDataset(
            sequences=self.sequences,
            beliefs=beliefs,
            probs=jnp.asarray(self.probs),
            activations_by_layer={k: jnp.asarray(v) for k, v in self.activations_by_layer.items()},
        )


class DeduplicationCache:
    """Cache for deduplicated datasets with optional disk persistence."""

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize the cache.

        Args:
            cache_dir: Optional directory for disk persistence. If None, only in-memory caching is used.
        """
        self._memory_cache: dict[str, DeduplicatedDataset] = {}
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path | None:
        """Get the disk cache path for a given key."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.pkl"

    def get(self, key: str) -> DeduplicatedDataset | None:
        """Retrieve a cached dataset by key.

        Checks memory first, then disk if cache_dir was configured.

        Args:
            key: The cache key

        Returns:
            The cached DeduplicatedDataset, or None if not found
        """
        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_path = self._get_cache_path(key)
        if cache_path and cache_path.exists():
            with cache_path.open("rb") as f:
                serializable = pickle.load(f)
            dataset = serializable.to_deduplicated_dataset()
            self._memory_cache[key] = dataset
            return dataset

        return None

    def put(self, key: str, dataset: DeduplicatedDataset, persist: bool = True) -> None:
        """Store a dataset in the cache.

        Args:
            key: The cache key
            dataset: The deduplicated dataset to cache
            persist: If True and cache_dir is configured, also save to disk
        """
        self._memory_cache[key] = dataset

        cache_path = self._get_cache_path(key)
        if persist and cache_path:
            serializable = SerializableDeduplicatedDataset.from_deduplicated_dataset(dataset)
            with cache_path.open("wb") as f:
                pickle.dump(serializable, f)

    def clear(self, key: str | None = None) -> None:
        """Clear cached entries.

        Args:
            key: If provided, clear only that key. Otherwise, clear all entries.
        """
        if key is None:
            self._memory_cache.clear()
            if self._cache_dir:
                for cache_file in self._cache_dir.glob("*.pkl"):
                    cache_file.unlink()
        else:
            self._memory_cache.pop(key, None)
            cache_path = self._get_cache_path(key)
            if cache_path and cache_path.exists():
                cache_path.unlink()

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if key in self._memory_cache:
            return True
        cache_path = self._get_cache_path(key)
        return cache_path is not None and cache_path.exists()
