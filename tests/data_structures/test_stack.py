import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.data_structures.stack import Stack


class Element(eqx.Module):
    x: jax.Array
    y: jax.Array
    z: jax.Array


@pytest.fixture
def default_element() -> Element:
    return Element(x=jnp.zeros(4), y=jnp.zeros((2, 3)), z=jnp.array(0))


@pytest.fixture
def stack(default_element: Element) -> Stack:
    return Stack(max_size=2, default_element=default_element)


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


def test_push(stack: Stack, elements: list[Element]):
    """Test basic push operation."""
    stack = stack.push(elements[0])
    assert stack.size == 1
    assert not stack.is_full
    assert not stack.is_empty

    stack = stack.push(elements[1])

    assert stack.size == 2
    assert stack.is_full
    assert not stack.is_empty


def test_full_push(stack: Stack, elements: list[Element]):
    """Test adding to full data structure."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])
    stack = stack.push(elements[2])  # Should not add

    assert stack.size == 2
    assert stack.is_full
    assert not stack.is_empty


def test_pop(stack: Stack, elements: list[Element]):
    """Test basic pop operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])

    stack, val = stack.pop()
    assert stack.size == 1
    assert isinstance(val, Element)
    chex.assert_trees_all_equal(val, elements[1])

    stack, val = stack.pop()
    assert stack.size == 0
    assert isinstance(val, Element)
    chex.assert_trees_all_equal(val, elements[0])


def test_empty_pop(stack: Stack, default_element: Element):
    """Test removing from empty data structure."""
    stack, val = stack.pop()
    chex.assert_trees_all_equal(val, default_element)
    assert stack.is_empty


def test_peek(stack: Stack, elements: list[Element]):
    """Test peek operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])

    val = stack.peek()
    assert stack.size == 2  # Size unchanged
    chex.assert_trees_all_equal(val, elements[1])


def test_empty_peek(stack: Stack, default_element: Element):
    """Test peeking empty data structure."""
    val = stack.peek()
    chex.assert_trees_all_equal(val, default_element)


def test_clear(stack: Stack, elements: list[Element]):
    """Test clear operation."""
    stack = stack.push(elements[0])
    stack = stack.push(elements[1])
    stack = stack.clear()

    assert stack.is_empty
