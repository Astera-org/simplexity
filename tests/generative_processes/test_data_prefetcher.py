"""Test the data prefetcher module."""

import threading
import time

import pytest

from simplexity.generative_processes.data_prefetcher import DataPrefetcher


def test_get_returns_correct_result():
    """Get should return the result of generate_fn called with the step number."""
    prefetcher = DataPrefetcher(lambda step: step * 10, lookahead=1)
    assert prefetcher.get(0) == 0
    assert prefetcher.get(5) == 50
    assert prefetcher.get(100) == 1000
    prefetcher.shutdown()


def test_prefetch_submits_future():
    """Prefetch should submit a future that can be retrieved by get without re-computation."""
    call_count = 0

    def counting_fn(step: int) -> int:
        nonlocal call_count
        call_count += 1
        return step

    prefetcher = DataPrefetcher(counting_fn, lookahead=1)
    prefetcher.prefetch(0)
    result = prefetcher.get(0)
    assert result == 0
    assert call_count == 1
    prefetcher.shutdown()


def test_lookahead_prefetches_future_steps():
    """Get should trigger prefetch for the next lookahead steps."""
    called_steps: list[int] = []

    def tracking_fn(step: int) -> int:
        called_steps.append(step)
        return step

    prefetcher = DataPrefetcher(tracking_fn, lookahead=2)
    prefetcher.get(0)
    prefetcher.get(1)
    prefetcher.get(2)
    assert sorted(set(called_steps)) == [0, 1, 2]
    assert called_steps.count(0) == 1
    assert called_steps.count(1) == 1
    assert called_steps.count(2) == 1
    prefetcher.shutdown()


def test_get_cleans_up_old_futures():
    """Get should not hold references to old step results after advancing."""
    call_count = 0

    def counting_fn(step: int) -> int:
        nonlocal call_count
        call_count += 1
        return step

    prefetcher = DataPrefetcher(counting_fn, lookahead=1)
    prefetcher.prefetch(0)
    prefetcher.prefetch(1)
    prefetcher.get(2)
    old_count = call_count
    prefetcher.get(0)
    assert call_count > old_count
    prefetcher.shutdown()


def test_error_propagation():
    """Exceptions in generate_fn should be re-raised by get."""

    def failing_fn(step: int) -> int:
        raise ValueError(f"step {step} failed")

    prefetcher = DataPrefetcher(failing_fn, lookahead=1)
    with pytest.raises(ValueError, match="step 3 failed"):
        prefetcher.get(3)
    prefetcher.shutdown()


def test_shutdown_does_not_hang():
    """Shutdown should return promptly even with pending futures."""

    def slow_fn(step: int) -> int:
        time.sleep(10)
        return step

    prefetcher = DataPrefetcher(slow_fn, lookahead=1)
    prefetcher.prefetch(0)
    prefetcher.shutdown()


def test_generate_fn_runs_in_background_thread():
    """Generate function should execute in a different thread than the caller."""
    caller_thread = threading.current_thread().ident
    gen_thread_id: int | None = None

    def capture_thread(step: int) -> int:
        nonlocal gen_thread_id
        gen_thread_id = threading.current_thread().ident
        return step

    prefetcher = DataPrefetcher(capture_thread, lookahead=1)
    prefetcher.get(0)
    assert gen_thread_id is not None
    assert gen_thread_id != caller_thread
    prefetcher.shutdown()


def test_duplicate_prefetch_is_noop():
    """Calling prefetch twice for the same step should not submit a second task."""
    call_count = 0

    def counting_fn(step: int) -> int:
        nonlocal call_count
        call_count += 1
        return step

    prefetcher = DataPrefetcher(counting_fn, lookahead=1)
    prefetcher.prefetch(0)
    prefetcher.prefetch(0)
    prefetcher.get(0)
    assert call_count == 1
    prefetcher.shutdown()
