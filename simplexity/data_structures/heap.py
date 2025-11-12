"""Heap data structure."""

from collections.abc import Callable
from typing import Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.data_structures.collection import Element
from simplexity.data_structures.stack import Stack


class ArrayLike(Protocol):  # pylint: disable=too-few-public-methods
    """A protocol for array-like objects."""

    shape: tuple[int, ...]
    dtype: jnp.dtype


class Heap(Stack[Element]):
    """Generic heap for any PyTree structure."""

    compare: Callable[[Element, Element], jax.Array]

    def __init__(self, max_size: int, default_element: Element, compare: Callable[[Element, Element], jax.Array]):  # pylint: disable=super-init-not-called
        """Initialize empty queue/stack."""
        self.default_element = default_element

        def init_data(element: ArrayLike) -> jax.Array:
            return jnp.zeros((max_size,) + element.shape, dtype=element.dtype)

        self.data = jax.tree.map(init_data, default_element)
        self._size = jnp.array(0, dtype=jnp.int32)
        self.max_size = jnp.array(max_size, dtype=jnp.int32)
        self.compare = compare

    @eqx.filter_jit
    def push(self, element: Element) -> "Heap[Element]":
        """Push a new element onto the stack."""
        heap = super().push(element)
        heap = cast(Heap[Element], heap)
        heap = heap._bubble_up(heap.size - 1)  # pylint: disable=protected-access
        return heap

    @eqx.filter_jit
    def pop(self) -> tuple["Heap[Element]", Element]:
        """Pop the next element from the stack."""
        heap = self._swap(jnp.array(0, dtype=jnp.int32), self.size - 1)
        heap, element = super(Heap, heap).pop()
        heap = cast(Heap[Element], heap)
        heap = heap._bubble_down(jnp.array(0, dtype=jnp.int32))  # pylint: disable=protected-access
        return heap, element

    @eqx.filter_jit
    def peek(self) -> Element:
        """Look at the next element without removing it."""

        def do_nothing(heap: Heap) -> Element:
            return heap.default_element

        def do_peek(heap: Heap) -> Element:
            return jax.tree.map(lambda x: x[0], heap.data)

        return jax.lax.cond(self.is_empty, do_nothing, do_peek, self)

    @eqx.filter_jit
    def parent_idx(self, child_idx: jax.Array) -> jax.Array:
        """Get the parent of an element in the heap."""
        return (child_idx - 1) // 2

    @eqx.filter_jit
    def left_child_idx(self, parent_idx: jax.Array) -> jax.Array:
        """Get the left child of an element in the heap."""
        return 2 * parent_idx + 1

    @eqx.filter_jit
    def right_child_idx(self, parent_idx: jax.Array) -> jax.Array:
        """Get the right child of an element in the heap."""
        return 2 * parent_idx + 2

    @eqx.filter_jit
    def __getitem__(self, idx: jax.Array) -> Element:
        """Get an element from the heap."""
        return jax.tree.map(lambda x: x[idx], self.data)

    @eqx.filter_jit
    def _swap(self, index1: jax.Array, index2: jax.Array) -> "Heap[Element]":
        """Swap two elements in the heap."""
        elem1 = self[index1]
        elem2 = self[index2]
        heap = eqx.tree_at(lambda x: x.data[index1], self, elem2)
        heap = eqx.tree_at(lambda x: x.data[index2], heap, elem1)
        return heap

    @eqx.filter_jit
    def _bubble_up(self, child_idx: jax.Array) -> "Heap[Element]":
        """Bubble up the last element in the heap."""
        parent_idx = self.parent_idx(child_idx)
        parent = self[parent_idx]
        child = self[child_idx]

        def do_nothing(heap: "Heap[Element]") -> "Heap[Element]":
            return heap

        def do_bubble_up(heap: "Heap[Element]") -> "Heap[Element]":
            heap = self._swap(child_idx, parent_idx)
            return heap._bubble_up(parent_idx)  # pylint: disable=protected-access

        return jax.lax.cond(
            self.compare(child, parent) > 0,
            do_bubble_up,
            do_nothing,
        )

    @eqx.filter_jit
    def _bubble_down(self, parent_idx: jax.Array) -> "Heap[Element]":
        """Bubble down the first element in the heap."""
        left_child_idx = self.left_child_idx(parent_idx)
        right_child_idx = self.right_child_idx(parent_idx)
        parent = self[parent_idx]
        left_child = self[left_child_idx]
        right_child = self[right_child_idx]

        def do_nothing(heap: "Heap[Element]") -> "Heap[Element]":
            return heap

        def do_left_bubble_down(heap: "Heap[Element]") -> "Heap[Element]":
            heap = self._swap(parent_idx, left_child_idx)
            return heap._bubble_down(left_child_idx)  # pylint: disable=protected-access

        def do_right_bubble_down(heap: "Heap[Element]") -> "Heap[Element]":
            heap = self._swap(parent_idx, right_child_idx)
            return heap._bubble_down(right_child_idx)  # pylint: disable=protected-access

        return jax.lax.cond(
            self.compare(parent, left_child) > 0,
            lambda: jax.lax.cond(
                self.compare(left_child, right_child) > 0,
                do_left_bubble_down,
                do_right_bubble_down,
            ),
            do_nothing,
        )
