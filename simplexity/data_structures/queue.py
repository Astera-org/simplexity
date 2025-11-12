"""Queue data structure."""

import equinox as eqx
import jax

from simplexity.data_structures.collection import Collection, Element
from simplexity.data_structures.stack import Stack


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
                return jax.tree.map(lambda x: x[0], stack.data)

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
