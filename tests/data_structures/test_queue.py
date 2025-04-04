import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.data_structures.queue import Queue


class Element(eqx.Module):
    x: jax.Array
    y: jax.Array
    z: jax.Array


@pytest.fixture
def default_element() -> Element:
    return Element(x=jnp.zeros(4), y=jnp.zeros((2, 3)), z=jnp.array(0))


@pytest.fixture
def queue(default_element: Element) -> Queue:
    return Queue(max_size=2, default_element=default_element)


@pytest.fixture
def elements() -> list[Element]:
    num_elements = 3
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3 * num_elements)
    return [
        Element(
            x=jax.random.uniform(keys[3 * i], (4,)),
            y=jax.random.uniform(keys[3 * i + 1], (2, 3)),
            z=jax.random.randint(keys[3 * i + 2], (), 0, 10),
        )
        for i in range(num_elements)
    ]


def test_add(queue: Queue, elements: list[Element]):
    """Test basic add operation."""
    queue = queue.enqueue(elements[0])
    assert queue.size == 1
    assert not queue.is_full
    assert not queue.is_empty

    queue = queue.enqueue(elements[1])

    assert queue.size == 2
    assert queue.is_full
    assert not queue.is_empty


def test_full_add(queue: Queue, elements: list[Element]):
    """Test adding to full data structure."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])
    queue = queue.enqueue(elements[2])  # Should not add

    assert queue.size == 2
    assert queue.is_full
    assert not queue.is_empty


def test_remove(queue: Queue, elements: list[Element]):
    """Test basic remove operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])

    queue, val = queue.dequeue()
    assert queue.size == 1
    assert isinstance(val, Element)
    chex.assert_trees_all_equal(val, elements[0])

    queue, val = queue.dequeue()
    assert queue.size == 0
    assert isinstance(val, Element)
    chex.assert_trees_all_equal(val, elements[1])


def test_empty_remove(queue: Queue, default_element: Element):
    """Test removing from empty data structure."""
    queue, val = queue.dequeue()
    chex.assert_trees_all_equal(val, default_element)
    assert queue.is_empty


def test_peek(queue: Queue, elements: list[Element]):
    """Test peek operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])

    val = queue.peek()
    assert queue.size == 2  # Size unchanged
    chex.assert_trees_all_equal(val, elements[0])


def test_empty_peek(queue: Queue, default_element: Element):
    """Test peeking empty data structure."""
    val = queue.peek()
    chex.assert_trees_all_equal(val, default_element)


def test_clear(queue: Queue, elements: list[Element]):
    """Test clear operation."""
    queue = queue.enqueue(elements[0])
    queue = queue.enqueue(elements[1])
    queue = queue.clear()

    assert queue.is_empty
