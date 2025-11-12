"""Collection data structure."""

from abc import abstractmethod
from typing import TypeVar

import equinox as eqx
import jax

Element = TypeVar("Element")


class Collection[Element](eqx.Module):
    """Generic collection for any PyTree structure."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def size(self) -> jax.Array:
        """The number of elements in the collection."""

    @property
    @abstractmethod
    def is_empty(self) -> jax.Array:
        """Whether the collection is empty."""

    @property
    @abstractmethod
    def is_full(self) -> jax.Array:
        """Whether the collection is full."""

    @abstractmethod
    def add(self, element: Element) -> "Collection[Element]":
        """Add an element to the collection."""

    @abstractmethod
    def remove(self) -> tuple["Collection[Element]", Element]:
        """Remove an element from the collection."""

    @abstractmethod
    def peek(self) -> Element:
        """Look at the next element without removing it."""

    @abstractmethod
    def clear(self) -> "Collection[Element]":
        """Clear the collection."""
