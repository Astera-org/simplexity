"""Stack data structure."""

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.data_structures.collection import Collection, Element


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
        self.data = jax.tree.map(lambda x: jnp.zeros((max_size,) + x.shape, dtype=x.dtype), default_element)
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
                (jax.tree.map(lambda arr, val: arr.at[stack.size].set(val), stack.data, element), stack.size + 1),
            )

        return jax.lax.cond(self.is_full, do_nothing, do_push, self)

    @eqx.filter_jit
    def pop(self) -> tuple["Stack[Element]", Element]:
        """Pop the next element from the stack."""

        def do_nothing(stack: Stack) -> tuple["Stack[Element]", Element]:
            return stack, self.default_element

        def do_pop(stack: Stack) -> tuple["Stack[Element]", Element]:
            element = jax.tree.map(lambda x: x[stack.size - 1], stack.data)
            stack = eqx.tree_at(lambda s: s.size, stack, stack.size - 1)
            return stack, element

        return jax.lax.cond(self.is_empty, do_nothing, do_pop, self)

    @eqx.filter_jit
    def peek(self) -> Element:
        """Look at the next element without removing it."""

        def do_nothing(stack: Stack) -> Element:
            return stack.default_element

        def do_peek(stack: Stack) -> Element:
            return jax.tree.map(lambda x: x[stack.size - 1], stack.data)

        return jax.lax.cond(self.is_empty, do_nothing, do_peek, self)

    @eqx.filter_jit
    def clear(self) -> "Stack[Element]":
        """Clear the stack."""
        return eqx.tree_at(lambda s: s.size, self, 0)

    add = push
    remove = pop
