import chex
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.data_structures import Queue, Stack


@pytest.fixture
def stack() -> Stack:
    return Stack(max_size=2, default_element=jnp.zeros(2))


@pytest.fixture
def queue() -> Queue:
    return Queue(max_size=2, default_element=jnp.zeros(2))


@pytest.fixture
def elements() -> jax.Array:
    key = jax.random.PRNGKey(0)
    return jax.random.uniform(key, (3, 2))


def test_stack_push(stack: Stack, elements: jax.Array):
    """Test basic push operation."""
    stack = stack.push(elements[0])
    assert stack.size == 1
    assert not stack.is_full
    assert not stack.is_empty

    stack = stack.push(elements[1])

    assert stack.size == 2
    assert stack.is_full
    assert not stack.is_empty

    chex.assert_trees_all_equal(stack.data[0], elements[0])
    chex.assert_trees_all_equal(stack.data[1], elements[1])


def test_stack_full_push(stack: Stack, elements: jax.Array):
    """Test pushing to a full stack."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])
    stack = stack.push(elements[2])  # Should not add

    assert stack.size == 2
    assert stack.is_full
    assert not stack.is_empty

    chex.assert_trees_all_equal(stack.data[0], elements[0])
    chex.assert_trees_all_equal(stack.data[1], elements[1])


def test_stack_pop(stack: Stack, elements: jax.Array):
    """Test basic pop operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])

    stack, val = stack.pop()
    assert stack.size == 1
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, elements[1])

    stack, val = stack.pop()
    assert stack.size == 0
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, elements[0])


def test_stack_empty_pop(stack: Stack):
    """Test popping from empty stack."""
    stack, val = stack.pop()
    assert jnp.all(val == stack.default_element)
    assert stack.is_empty


def test_stack_peek(stack: Stack, elements: jax.Array):
    """Test peek operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])

    val = stack.peek()
    assert stack.size == 2  # Size unchanged
    chex.assert_trees_all_equal(val, elements[1])


def test_stack_empty_peek(stack: Stack):
    """Test peeking empty stack."""
    val = stack.peek()
    assert jnp.all(val == stack.default_element)


def test_stack_clear(stack: Stack, elements: jax.Array):
    """Test clear operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])
    stack = stack.clear()

    assert stack.is_empty


def test_queue_enqueue(queue: Queue, elements: jax.Array):
    """Test basic enqueue operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])

    assert queue.size == 2
    assert not queue.is_empty
    assert not queue.is_full


def test_queue_full_enqueue(queue: Queue, elements: jax.Array):
    """Test enqueueing to full queue."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])
    queue = queue.enqueue(elements[2])  # Should not add

    assert queue.size == 2
    assert queue.is_full
    assert not queue.is_empty


def test_queue_dequeue(queue: Queue, elements: jax.Array):
    """Test dequeue operation with restacking."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])

    queue, val = queue.dequeue()  # Should restack
    queue = queue.enqueue(elements[2])

    queue, val = queue.dequeue()  # Should restack
    assert queue.size == 1
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, elements[1])

    queue, val = queue.dequeue()  # Should not restack
    assert queue.size == 0
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, elements[2])


def test_queue_empty_dequeue(queue: Queue):
    """Test dequeuing from empty queue."""
    queue, val = queue.dequeue()
    assert jnp.all(val == queue.default_element)
    assert queue.is_empty


def test_queue_peek(queue: Queue, elements: jax.Array):
    """Test peek operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])

    val = queue.peek()
    assert isinstance(val, jax.Array)
    assert queue.size == 2
    chex.assert_trees_all_equal(val, elements[0])

    queue, _ = queue.dequeue()
    val = queue.peek()
    assert isinstance(val, jax.Array)
    assert queue.size == 1
    chex.assert_trees_all_equal(val, elements[1])


def test_queue_empty_peek(queue: Queue):
    """Test peeking empty queue."""
    val = queue.peek()
    assert jnp.all(val == queue.default_element)


def test_queue_clear(queue: Queue, elements: jax.Array):
    """Test clear operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])
    queue = queue.clear()

    assert queue.is_empty
