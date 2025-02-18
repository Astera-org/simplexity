from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp

from simplexity.generative_processes.data_structures import Collection, Queue, Stack
from simplexity.generative_processes.generalized_hidden_markov_model import GeneralizedHiddenMarkovModel

NodeDict = dict[tuple[int, ...], float]


class MixedStateNode(eqx.Module):
    """A node in a mixed state presentation of a generative process."""

    sequence: jax.Array
    sequence_length: jax.Array
    log_state: jax.Array
    log_probability: jax.Array

    @property
    def max_sequence_length(self) -> int:
        """The maximum length of the sequence."""
        return self.sequence.shape[0]


class TreeData(eqx.Module):
    """Data for a tree."""

    sequences: jax.Array
    sequence_lengths: jax.Array
    log_probabilities: jax.Array
    size: jax.Array

    @classmethod
    def empty(cls, max_size: int, max_sequence_length: int) -> "TreeData":
        """Create an empty tree."""
        return cls(
            sequences=jnp.zeros((max_size, max_sequence_length), dtype=jnp.int32),
            sequence_lengths=jnp.zeros((max_size,), dtype=jnp.int32),
            log_probabilities=jnp.zeros((max_size,), dtype=jnp.float32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def add(self, node: MixedStateNode) -> "TreeData":
        """Add a sequence to the tree."""
        return TreeData(
            sequences=self.sequences.at[self.size].set(node.sequence),
            sequence_lengths=self.sequence_lengths.at[self.size].set(node.sequence_length),
            log_probabilities=self.log_probabilities.at[self.size].set(node.log_probability),
            size=self.size + 1,
        )

    @property
    def max_size(self) -> int:
        """The maximum number of elements in the tree."""
        return self.sequences.shape[0]

    @property
    def max_sequence_length(self) -> int:
        """The maximum length of the sequences."""
        return self.sequences.shape[1]


class MixedStateTree:
    """A presentation of a generative process as a mixed state."""

    def __init__(self, nodes: TreeData):
        self.nodes: NodeDict = {}
        for i in range(nodes.size):
            self.add(nodes.sequences[i, : nodes.sequence_lengths[i]], nodes.log_probabilities[i])

    def __len__(self) -> int:
        """The number of nodes in the tree."""
        return len(self.nodes)

    def add(self, sequence: jax.Array, log_probability: jax.Array) -> None:
        """Add a sequence to the tree."""
        key = tuple(sequence.tolist())
        value = log_probability.item()
        self.nodes[key] = value


class SearchAlgorithm(Enum):
    """The algorithm to use for searching the tree."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"


class MixedStateTreeGenerator(eqx.Module):
    """A generator of nodes in a mixed state presentation of a generative process."""

    ghmm: GeneralizedHiddenMarkovModel
    max_sequence_length: int
    max_tree_size: int
    log_prob_threshold: float

    def __init__(
        self,
        ghmm: GeneralizedHiddenMarkovModel,
        max_sequence_length: int,
        max_tree_size: int,
        log_prob_threshold: float = -jnp.inf,
    ):
        self.ghmm = ghmm
        self.max_sequence_length = max_sequence_length
        self.max_tree_size = max_tree_size
        self.log_prob_threshold = log_prob_threshold

    def generate(self, search_algorithm: SearchAlgorithm = SearchAlgorithm.DEPTH_FIRST) -> MixedStateTree:
        """Generate all nodes in the tree."""

        def continue_loop(carry: tuple[TreeData, Collection[MixedStateNode]]) -> jax.Array:
            tree_data, search_nodes = carry
            return jnp.logical_and(~search_nodes.is_empty, tree_data.size < tree_data.max_size)

        def add_next_node(
            carry: tuple[TreeData, Collection[MixedStateNode]],
        ) -> tuple[TreeData, Collection[MixedStateNode]]:
            tree_data, search_nodes = carry
            search_nodes, node = self._next_node(search_nodes)
            tree_data = tree_data.add(node)
            return tree_data, search_nodes

        tree_data = TreeData.empty(self.max_tree_size, self.max_sequence_length)
        if search_algorithm == SearchAlgorithm.BREADTH_FIRST:
            search_nodes = Queue(self.max_tree_size, default_element=self.root)
        else:
            search_nodes = Stack(self.max_tree_size, default_element=self.root)
        search_nodes = search_nodes.add(self.root)

        tree_data, _ = jax.lax.while_loop(continue_loop, add_next_node, (tree_data, search_nodes))
        return MixedStateTree(tree_data)

    @property
    def root(self) -> MixedStateNode:
        """The root node of the tree."""
        empty_sequence = jnp.zeros((self.max_sequence_length,), dtype=jnp.int32)
        sequence_length = jnp.array(0)
        log_state = self.ghmm.log_right_eigenvector
        log_probability = jax.nn.logsumexp(self.ghmm.log_left_eigenvector + log_state)
        return MixedStateNode(empty_sequence, sequence_length, log_state, log_probability)

    @eqx.filter_jit
    def get_child(self, node: MixedStateNode, obs: jax.Array) -> MixedStateNode:
        """Get the child of a node."""
        sequence = node.sequence.at[node.sequence_length].set(obs)
        sequence_length = node.sequence_length + 1
        log_state = jax.nn.logsumexp(self.ghmm.log_transition_matrices[obs] + node.log_state, axis=1)
        log_probability = jax.nn.logsumexp(self.ghmm.log_left_eigenvector + log_state)
        return MixedStateNode(sequence, sequence_length, log_state, log_probability)

    @eqx.filter_jit
    def _next_node(self, nodes: Collection[MixedStateNode]) -> tuple[Collection[MixedStateNode], MixedStateNode]:
        """Get the next node from a collection and add that node's children to the collection."""

        def add_children(
            nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
        ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
            def maybe_add_child(
                i: int, nodes_node: tuple[Collection[MixedStateNode], MixedStateNode]
            ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
                nodes, node = nodes_node
                obs = jnp.array(i)
                child = self.get_child(node, obs)

                def add_child(
                    nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
                ) -> Collection[MixedStateNode]:
                    nodes, node = nodes_node
                    return nodes.add(node)

                def do_nothing(
                    nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
                ) -> Collection[MixedStateNode]:
                    nodes, _ = nodes_node
                    return nodes

                nodes = jax.lax.cond(
                    child.log_probability >= self.log_prob_threshold, add_child, do_nothing, (nodes, child)
                )
                return nodes, node

            return jax.lax.fori_loop(0, self.ghmm.num_observations, maybe_add_child, nodes_node)

        def no_update(
            nodes_node: tuple[Collection[MixedStateNode], MixedStateNode],
        ) -> tuple[Collection[MixedStateNode], MixedStateNode]:
            return nodes_node

        nodes, node = nodes.remove()
        return jax.lax.cond(node.sequence_length < node.max_sequence_length, add_children, no_update, (nodes, node))
