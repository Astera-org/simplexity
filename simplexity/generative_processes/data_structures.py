from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

Element = TypeVar("Element")


class Collection(eqx.Module, Generic[Element]):
    """Generic collection for any PyTree structure."""

    @property
    @abstractmethod
    def is_empty(self) -> jax.Array:
        """Whether the collection is empty."""
        ...

    @abstractmethod
    def add(self, element: Element) -> "Collection[Element]":
        """Add an element to the collection."""
        ...

    @abstractmethod
    def remove(self) -> tuple["Collection[Element]", Element]:
        """Remove an element from the collection."""
        ...


class Stack(Collection[Element]):
    """Generic stack for any PyTree structure."""

    default_element: Element
    data: jax.Array  # stores PyTree structures
    size: jax.Array

    @property
    def max_size(self) -> int:
        """The maximum number of elements that can be stored in the queue/stack."""
        return self.data.shape[0]

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
        self.size = jnp.array(0, dtype=jnp.int32)

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
            return stack, jax.tree_map(lambda x: jnp.zeros_like(x[0]), stack.data)

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
