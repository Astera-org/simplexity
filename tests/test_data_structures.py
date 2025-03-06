import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from simplexity.generative_processes.data_structures import Collection, Queue, Stack


class TestElement(eqx.Module):
    x: jax.Array
    y: jax.Array
    z: jax.Array


@pytest.fixture
def default_element() -> TestElement:
    return TestElement(x=jnp.zeros(4), y=jnp.zeros((2, 3)), z=jnp.array(0))


@pytest.fixture
def stack(default_element: TestElement) -> Stack:
    return Stack(max_size=2, default_element=default_element)


@pytest.fixture
def queue(default_element: TestElement) -> Queue:
    return Queue(max_size=2, default_element=default_element)


@pytest.fixture
def elements() -> list[TestElement]:
    num_elements = 3
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3 * num_elements)
    return [
        TestElement(
            x=jax.random.uniform(keys[3 * i], (4,)),
            y=jax.random.uniform(keys[3 * i + 1], (2, 3)),
            z=jax.random.randint(keys[3 * i + 2], (), 0, 10),
        )
        for i in range(num_elements)
    ]


@pytest.mark.parametrize("data_structure_name", ["stack", "queue"])
def test_add(data_structure_name: str, elements: jax.Array, request: pytest.FixtureRequest):
    """Test basic add operation."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure = data_structure.add(elements[0])
    assert data_structure.size == 1
    assert not data_structure.is_full
    assert not data_structure.is_empty

    data_structure = data_structure.add(elements[1])

    assert data_structure.size == 2
    assert data_structure.is_full
    assert not data_structure.is_empty


@pytest.mark.parametrize("data_structure_name", ["stack", "queue"])
def test_full_add(data_structure_name: str, elements: jax.Array, request: pytest.FixtureRequest):
    """Test adding to full data structure."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure = data_structure.add(elements[0])
    data_structure = data_structure.add(elements[1])
    data_structure = data_structure.add(elements[2])  # Should not add

    assert data_structure.size == 2
    assert data_structure.is_full
    assert not data_structure.is_empty


@pytest.mark.parametrize(("data_structure_name", "remove_order"), [("stack", (1, 0)), ("queue", (0, 1))])
def test_remove(
    data_structure_name: str, remove_order: tuple[int, int], elements: jax.Array, request: pytest.FixtureRequest
):
    """Test basic remove operation."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure = data_structure.add(elements[0])
    data_structure = data_structure.add(elements[1])

    data_structure, val = data_structure.remove()
    assert data_structure.size == 1
    assert isinstance(val, TestElement)
    chex.assert_trees_all_equal(val, elements[remove_order[0]])

    data_structure, val = data_structure.remove()
    assert data_structure.size == 0
    assert isinstance(val, TestElement)
    chex.assert_trees_all_equal(val, elements[remove_order[1]])


@pytest.mark.parametrize("data_structure_name", ["stack", "queue"])
def test_empty_remove(data_structure_name: str, default_element: jax.Array, request: pytest.FixtureRequest):
    """Test removing from empty data structure."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure, val = data_structure.remove()
    chex.assert_trees_all_equal(val, default_element)
    assert data_structure.is_empty


@pytest.mark.parametrize(("data_structure_name", "peek_idx"), [("stack", 1), ("queue", 0)])
def test_peek(data_structure_name: str, peek_idx: int, elements: jax.Array, request: pytest.FixtureRequest):
    """Test peek operation."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure = data_structure.add(elements[0])
    data_structure = data_structure.add(elements[1])

    val = data_structure.peek()
    assert data_structure.size == 2  # Size unchanged
    chex.assert_trees_all_equal(val, elements[peek_idx])


@pytest.mark.parametrize("data_structure_name", ["stack", "queue"])
def test_empty_peek(data_structure_name: str, default_element: jax.Array, request: pytest.FixtureRequest):
    """Test peeking empty data structure."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    val = data_structure.peek()
    chex.assert_trees_all_equal(val, default_element)


@pytest.mark.parametrize("data_structure_name", ["stack", "queue"])
def test_clear(data_structure_name: str, elements: jax.Array, request: pytest.FixtureRequest):
    """Test clear operation."""
    data_structure: Collection = request.getfixturevalue(data_structure_name)
    data_structure = data_structure.add(elements[0])
    data_structure = data_structure.add(elements[1])
    data_structure = data_structure.clear()

    assert data_structure.is_empty
