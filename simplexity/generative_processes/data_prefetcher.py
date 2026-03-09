"""Background data prefetcher for overlapping data generation with training."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType


class DataPrefetcher[T]:
    """Prefetches training data in background threads to overlap with GPU training.

    Uses a thread pool to generate upcoming batches while the current batch is being trained on.
    This works because JAX JIT-compiled functions release the GIL, enabling genuine parallelism
    between JAX data generation and PyTorch training.

    Not thread-safe: ``prefetch``, ``get``, and ``shutdown`` should all be called from a single
    thread. The background thread pool is managed internally.

    Intended to be used as a context manager::

        with DataPrefetcher(generate_fn, lookahead=1) as prefetcher:
            for step in range(num_steps):
                batch = prefetcher.get(step)
                train(batch)

    Args:
        generate_fn: A function that takes a step number (int) and returns batch data.
        lookahead: Number of future steps to prefetch. Defaults to 1.
    """

    def __init__(self, generate_fn: Callable[[int], T], lookahead: int = 1) -> None:
        self._generate_fn = generate_fn
        self._lookahead = lookahead
        self._executor = ThreadPoolExecutor(max_workers=lookahead + 1)
        self._futures: dict[int, Future[T]] = {}

    def __enter__(self) -> "DataPrefetcher[T]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    def prefetch(self, step: int) -> None:
        """Submit a background task to generate data for the given step."""
        if step not in self._futures:
            self._futures[step] = self._executor.submit(self._generate_fn, step)

    def get(self, step: int) -> T:
        """Return the generated data for the given step, blocking until ready.

        Also triggers prefetch for the next `lookahead` steps and cleans up old futures.
        """
        self.prefetch(step)
        for s in range(step + 1, step + 1 + self._lookahead):
            self.prefetch(s)
        result = self._futures.pop(step).result()
        old_keys = [k for k in self._futures if k < step]
        for k in old_keys:
            self._futures.pop(k).cancel()
        return result

    def shutdown(self) -> None:
        """Cancel pending futures and shut down the thread pool."""
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
