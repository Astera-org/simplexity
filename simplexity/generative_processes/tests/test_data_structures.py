import chex
import jax
import jax.numpy as jnp

from simplexity.generative_processes.data_structures import Queue, Stack


def test_stack_push():
    """Test basic push operation."""
    stack = Stack(max_size=3, default_element=jnp.zeros(2))

    element_1 = jnp.array([1.0, 2.0])
    element_2 = jnp.array([3.0, 4.0])

    stack = stack.push(element_1)
    stack = stack.push(element_2)

    assert stack.size == 2
    assert not stack.is_full
    assert not stack.is_empty

    chex.assert_trees_all_equal(stack.data[0], element_1)
    chex.assert_trees_all_equal(stack.data[1], element_2)


def test_stack_full_push():
    """Test pushing to a full stack."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    element_1 = jnp.array([1.0])
    element_2 = jnp.array([2.0])
    element_3 = jnp.array([3.0])

    stack = stack.push(element_1)
    stack = stack.push(element_2)
    stack = stack.push(element_3)  # Should not add

    assert stack.size == 2
    assert stack.is_full
    assert not stack.is_empty

    chex.assert_trees_all_equal(stack.data[0], element_1)
    chex.assert_trees_all_equal(stack.data[1], element_2)


def test_stack_pop():
    """Test basic pop operation."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    element_1 = jnp.array([1.0])
    element_2 = jnp.array([2.0])

    stack = stack.push(element_1)
    stack = stack.push(element_2)

    stack, val = stack.pop()
    assert stack.size == 1
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, element_2)


def test_stack_empty_pop():
    """Test popping from empty stack."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    stack, val = stack.pop()
    assert jnp.all(val == stack.default_element)
    assert stack.is_empty


def test_stack_peek():
    """Test peek operation."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    element = jnp.array([1.0])
    stack = stack.push(element)

    val = stack.peek()
    assert stack.size == 1  # Size unchanged
    chex.assert_trees_all_equal(val, element)


def test_stack_empty_peek():
    """Test peeking empty stack."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    val = stack.peek()
    assert jnp.all(val == stack.default_element)


def test_stack_clear():
    """Test clear operation."""
    stack = Stack(max_size=2, default_element=jnp.zeros(1))

    stack = stack.push(jnp.array([1.0]))
    stack = stack.clear()

    assert stack.is_empty


def test_queue_enqueue():
    """Test basic enqueue operation."""
    queue = Queue(max_size=3, default_element=jnp.zeros(1))

    element_1 = jnp.array([1.0])
    element_2 = jnp.array([2.0])

    queue = queue.enqueue(element_1)
    queue = queue.enqueue(element_2)

    assert queue.size == 2
    assert not queue.is_empty
    assert not queue.is_full


def test_queue_full_enqueue():
    """Test enqueueing to full queue."""
    queue = Queue(max_size=2, default_element=jnp.zeros(1))

    queue = queue.enqueue(jnp.array([1.0]))
    queue = queue.enqueue(jnp.array([2.0]))
    queue = queue.enqueue(jnp.array([3.0]))  # Should not add

    assert queue.size == 2
    assert queue.is_full
    assert not queue.is_empty


def test_queue_dequeue():
    """Test dequeue operation with restacking."""
    queue = Queue(max_size=3, default_element=jnp.zeros(1))

    element_1 = jnp.array([1.0])
    element_2 = jnp.array([2.0])

    queue = queue.enqueue(element_1)
    queue = queue.enqueue(element_2)

    queue, val = queue.dequeue()  # Should restack
    assert queue.size == 1
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, element_1)

    queue, val = queue.dequeue()  # Should not restack
    assert queue.size == 0
    assert isinstance(val, jax.Array)
    chex.assert_trees_all_equal(val, element_2)


def test_queue_empty_dequeue():
    """Test dequeuing from empty queue."""
    queue = Queue(max_size=2, default_element=jnp.zeros(1))

    queue, val = queue.dequeue()
    assert jnp.all(val == queue.default_element)
    assert queue.is_empty


def test_queue_peek():
    """Test peek operation."""
    queue = Queue(max_size=2, default_element=jnp.zeros(1))

    element_1 = jnp.array([1.0])
    element_2 = jnp.array([2.0])

    queue = queue.enqueue(element_1)
    queue = queue.enqueue(element_2)

    val = queue.peek()
    assert isinstance(val, jax.Array)
    assert queue.size == 2
    chex.assert_trees_all_equal(val, element_1)

    queue, _ = queue.dequeue()
    val = queue.peek()
    assert isinstance(val, jax.Array)
    assert queue.size == 1
    chex.assert_trees_all_equal(val, element_2)


def test_queue_empty_peek():
    """Test peeking empty queue."""
    queue = Queue(max_size=2, default_element=jnp.zeros(1))

    val = queue.peek()
    assert jnp.all(val == queue.default_element)


def test_queue_clear():
    """Test clear operation."""
    queue = Queue(max_size=2, default_element=jnp.zeros(1))

    queue = queue.enqueue(jnp.array([1.0]))
    queue = queue.clear()

    assert queue.is_empty
