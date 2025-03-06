from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp

Element = TypeVar("Element")


class Collection(eqx.Module, Generic[Element]):
    """Generic collection for any PyTree structure."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def size(self) -> jax.Array:
        """The number of elements in the collection."""
        ...

    @property
    @abstractmethod
    def is_empty(self) -> jax.Array:
        """Whether the collection is empty."""
        ...

    @property
    @abstractmethod
    def is_full(self) -> jax.Array:
        """Whether the collection is full."""
        ...

    @abstractmethod
    def add(self, element: Element) -> "Collection[Element]":
        """Add an element to the collection."""
        ...

    @abstractmethod
    def remove(self) -> tuple["Collection[Element]", Element]:
        """Remove an element from the collection."""
        ...

    @abstractmethod
    def peek(self) -> Element:
        """Look at the next element without removing it."""
        ...

    @abstractmethod
    def clear(self) -> "Collection[Element]":
        """Clear the collection."""
        ...


class Stack(Collection[Element]):
    """Generic stack for any PyTree structure."""

    default_element: Element
    data: jax.Array  # stores PyTree structures
    _size: jax.Array
    max_size: jax.Array

    @property
    def size(self) -> jax.Array:
        """The number of elements in the stack."""
        return self._size

    @property
    def is_empty(self) -> jax.Array:
        """Whether the stack is empty."""
        return self.size == 0

    @property
    def is_full(self) -> jax.Array:
        """Whether the stack is full."""
        return self.size == self.max_size

    def __init__(self, max_size: int, default_element: Element):
        """Initialize empty queue/stack."""
        self.default_element = default_element
        self.data = jax.tree_map(lambda x: jnp.zeros((max_size,) + x.shape, dtype=x.dtype), default_element)
        self._size = jnp.array(0, dtype=jnp.int32)
        self.max_size = jnp.array(max_size, dtype=jnp.int32)

    @eqx.filter_jit
    def push(self, element: Element) -> "Stack":
        """Push a new element onto the stack."""

        def do_nothing(stack: Stack) -> Stack:
            return stack

        def do_push(stack: Stack) -> Stack:
            return eqx.tree_at(
                lambda s: (s.data, s.size),
                stack,
                (jax.tree_map(lambda arr, val: arr.at[stack.size].set(val), stack.data, element), stack.size + 1),
            )

        return jax.lax.cond(self.is_full, do_nothing, do_push, self)

    @eqx.filter_jit
    def pop(self) -> tuple["Stack[Element]", Element]:
        """Pop the next element from the stack."""

        def do_nothing(stack: Stack) -> tuple["Stack[Element]", Element]:
            return stack, self.default_element

        def do_pop(stack: Stack) -> tuple["Stack[Element]", Element]:
            element = jax.tree_map(lambda x: x[stack.size - 1], stack.data)
            stack = eqx.tree_at(lambda s: s.size, stack, stack.size - 1)
            return stack, element

        return jax.lax.cond(self.is_empty, do_nothing, do_pop, self)

    @eqx.filter_jit
    def peek(self) -> Element:
        """Look at the next element without removing it."""

        def do_nothing(stack: Stack) -> Element:
            return stack.default_element

        def do_peek(stack: Stack) -> Element:
            return jax.tree_map(lambda x: x[stack.size - 1], stack.data)

        return jax.lax.cond(self.is_empty, do_nothing, do_peek, self)

    @eqx.filter_jit
    def clear(self) -> "Stack[Element]":
        """Clear the stack."""
        return eqx.tree_at(lambda s: s.size, self, 0)

    add = push
    remove = pop


class Queue(Collection[Element]):
    """Generic queue for any PyTree structure."""

    instack: Stack[Element]
    outstack: Stack[Element]

    def __init__(self, max_size: int, default_element: Element):
        """Initialize empty queue."""
        self.instack = Stack(max_size, default_element)
        self.outstack = Stack(max_size, default_element)

    @property
    def default_element(self) -> Element:
        """The default element for the queue."""
        return self.instack.default_element

    @property
    def data(self) -> jax.Array:
        """The data in the queue."""
        queue = self._restack()
        return queue.outstack.data

    @property
    def size(self) -> jax.Array:
        """The number of elements in the queue."""
        return self.instack.size + self.outstack.size

    @property
    def is_empty(self) -> jax.Array:
        """Whether the queue is empty."""
        return self.instack.is_empty & self.outstack.is_empty

    @property
    def is_full(self) -> jax.Array:
        """Whether the queue is full."""
        return self.instack.size + self.outstack.size >= self.instack.max_size

    @eqx.filter_jit
    def enqueue(self, element: Element) -> "Queue[Element]":
        """Add element to back of queue."""

        def do_nothing(queue: Queue) -> Queue:
            return queue

        def do_enqueue(queue: Queue) -> Queue:
            return eqx.tree_at(lambda q: q.instack, queue, queue.instack.push(element))

        return jax.lax.cond(self.is_full, do_nothing, do_enqueue, self)

    @eqx.filter_jit
    def dequeue(self) -> tuple["Queue[Element]", Element]:
        """Remove and return element from front of queue."""
        queue = self._restack()

        def do_nothing(stack: Stack) -> tuple[Stack[Element], Element]:
            return stack, stack.default_element

        def do_pop(stack: Stack[Element]) -> tuple[Stack[Element], Element]:
            return stack.pop()

        stack, element = jax.lax.cond(queue.outstack.is_empty, do_nothing, do_pop, queue.outstack)
        return eqx.tree_at(lambda q: q.outstack, queue, stack), element

    @eqx.filter_jit
    def peek(self) -> Element:
        """Look at front element without removing it."""

        def instack_peek(queue: Queue) -> Element:
            def empty(stack: Stack) -> Element:
                return stack.default_element

            def bottom(stack: Stack[Element]) -> Element:
                return jax.tree_map(lambda x: x[0], stack.data)

            return jax.lax.cond(queue.instack.is_empty, empty, bottom, queue.instack)

        def outstack_peek(queue: Queue) -> Element:
            return queue.outstack.peek()

        return jax.lax.cond(self.outstack.is_empty, instack_peek, outstack_peek, self)

    @eqx.filter_jit
    def clear(self) -> "Queue[Element]":
        """Clear the queue."""
        return eqx.tree_at(lambda q: (q.instack, q.outstack), self, (self.instack.clear(), self.outstack.clear()))

    @eqx.filter_jit
    def _restack(self) -> "Queue[Element]":
        """Restack the queue."""

        def transfer_elements(queue: "Queue[Element]") -> "Queue[Element]":
            """Transfer elements from instack to outstack."""

            def transfer_one(_, q):
                instack, val = q.instack.pop()
                outstack = q.outstack.push(val)
                return eqx.tree_at(lambda x: (x.instack, x.outstack), q, (instack, outstack))

            return jax.lax.fori_loop(0, queue.instack.size, transfer_one, queue)

        def do_nothing(queue: Queue) -> "Queue[Element]":
            return queue

        should_restack = self.outstack.is_empty & ~self.instack.is_empty
        return jax.lax.cond(should_restack, transfer_elements, do_nothing, self)

    add = enqueue
    remove = dequeue


class Heap(Stack[Element]):
    """Generic heap for any PyTree structure."""

    compare: Callable[[Element, Element], jax.Array]

    def __init__(self, max_size: int, default_element: Element, compare: Callable[[Element, Element], jax.Array]):
        """Initialize empty queue/stack."""
        self.default_element = default_element
        self.data = jax.tree_map(lambda x: jnp.zeros((max_size,) + x.shape, dtype=x.dtype), default_element)
        self._size = jnp.array(0, dtype=jnp.int32)
        self.max_size = jnp.array(max_size, dtype=jnp.int32)
        self.compare = compare

    @eqx.filter_jit
    def push(self, element: Element) -> "Heap[Element]":
        """Push a new element onto the stack."""
        heap = super().push(element)
        heap = cast(Heap[Element], heap)
        heap = heap._bubble_up(heap.size - 1)
        return heap

    @eqx.filter_jit
    def pop(self) -> tuple["Heap[Element]", Element]:
        """Pop the next element from the stack."""
        heap = self._swap(jnp.array(0, dtype=jnp.int32), self.size - 1)
        heap, element = super(Heap, heap).pop()
        heap = cast(Heap[Element], heap)
        heap = heap._bubble_down(jnp.array(0, dtype=jnp.int32))
        return heap, element

    @eqx.filter_jit
    def peek(self) -> Element:
        """Look at the next element without removing it."""

        def do_nothing(heap: Heap) -> Element:
            return heap.default_element

        def do_peek(heap: Heap) -> Element:
            return jax.tree_map(lambda x: x[0], heap.data)

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
        return jax.tree_map(lambda x: x[idx], self.data)

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
            return heap._bubble_up(parent_idx)

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
            return heap._bubble_down(left_child_idx)

        def do_right_bubble_down(heap: "Heap[Element]") -> "Heap[Element]":
            heap = self._swap(parent_idx, right_child_idx)
            return heap._bubble_down(right_child_idx)

        return jax.lax.cond(
            self.compare(parent, left_child) > 0,
            lambda: jax.lax.cond(
                self.compare(left_child, right_child) > 0,
                do_left_bubble_down,
                do_right_bubble_down,
            ),
            do_nothing,
        )
